# Structured output in LangChain: `@tool`-as-sink vs `with_structured_output`, and other methods

Five ways to get structured (JSON-shaped) data out of an LLM call. The
first two are used in this codebase, in `book_writer_commic_illustrated_agent.ipynb`;
the other three are documented for comparison — none are currently wired
into this notebook, but #3 (prebuilt agent + `response_format`) is the
best available fix for `researcher_node`'s `with_structured_output` con
(see Recommendation).

## `@tool`-as-sink (the pattern already used throughout this notebook)

Give the model a `@tool` whose *parameters* are the schema you want, run it
through a manual ReAct loop (`run_agent`), and capture the tool call's
validated args into a closure-captured dict instead of using the tool's
return value.

```python
def make_set_outline_tool(updates: dict):
    @tool
    def set_outline(pages: list[dict], summary: str) -> dict:
        """Set the page-by-page outline for the illustrated book.

        Args:
            pages: List of pages, each a dict with 'page_number' (int) and
                'topic' (str, a one-sentence scene description).
            summary: the description of the story
        """
        updates["PAGE_OUTLINE"] = pages
        updates["book_summary"] = summary
        return {"status": "success"}

    return set_outline

def outliner_node(state: BookState) -> dict:
    updates: dict = {}
    system_prompt = f"""
    Create a short illustrated story outline for: {state['PROMPT']}
    ...
    Use the 'set_outline' tool with your list of pages.
    """
    run_agent(system_prompt, [make_set_outline_tool(updates)])
    return updates
```

The structured data isn't the return value of the LLM call — it's a side
effect of the model calling the tool, captured by mutating `updates`.

### Pros

- Can mix freeform tool use (Wikipedia, Tavily, image gen) with structured
  capture in the *same* loop — `researcher_node` and `page_node`'s image
  step both need this: the model has to call an unknown number of search
  or image-gen tools before it has anything to structure.
- Works with any model `run_agent`/`bind_tools` supports, including ones
  with flaky native structured-output support.
- The `updates` dict pattern lets one node emit multiple different
  structured writes across several tool calls (e.g. `set_outline`, or
  `emphasis_node` calling `set_emphasis` once per page in a loop) without
  redefining one combined schema up front.
- Matches this codebase's established `@tool`-decorator style.

### Cons

- More boilerplate per node: define a tool, a closure dict, call
  `run_agent`, then read back out of the dict.
- No validation guarantee — the model can skip calling the tool and just
  reply with text. This is exactly what happened with the first version of
  `page_node_conversation`: it didn't use a tool at all, so `run_agent`
  returned free text wrapped in a ` ```json ` fence that had to be
  hand-parsed (`json.loads` + fence-stripping) before it was usable.
- Return value is threaded through a side effect (mutating `updates`),
  which is harder to trace/type than a normal return value — you only
  find out the actual shape by reading the tool's closure.
- Pays for a full ReAct loop (up to `max_tool_iterations` round trips)
  even when only one structured answer is needed — e.g. `cover_node`'s
  title prompt (`run_agent(title_prompt, [])`) pays for the loop
  machinery just to get one string back.

## `with_structured_output`

Bind a schema (TypedDict, Pydantic model, or JSON-schema dict) to the
model and call `.invoke()` — the return value *is* the parsed, validated
object. No tool, no closure, no manual JSON parsing.

```python
class PageConversationTurn(TypedDict):
    character_name: str
    speech: str

class PageConversationOutput(TypedDict):
    story_text: str
    conversation: list[PageConversationTurn]

page_conversation_model = model.with_structured_output(PageConversationOutput)

def page_node_conversation_wst(page_name: int, PROMPT: str, topic: str) -> dict:
    text_prompt = f"""
    Write the text and conversation for page {page_name} of a short illustrated story about: {PROMPT}

    PAGE TOPIC:
    {topic}

    Write 2-4 sentences of story text for this page.
    Basing on the text, infer a conversation for characters in the text.
    Limit each conversational turn to 1-2 sentences.
    """

    return page_conversation_model.invoke(text_prompt)
```

Note the prompt no longer needs the hand-written `<Output>`/`<Example>`
JSON-formatting instructions from the original — the schema itself
constrains the model's output shape.

### Pros

- One line, one round trip, returns the parsed/validated object directly.
- Stronger guarantee of getting the requested shape (provider-level JSON
  schema / forced tool call under the hood).
- Cleaner types: the return value is just `PageConversationOutput`, not
  "read whatever landed in a mutable dict."

### Cons

- Single-purpose per call — can't interleave other tool calls (Wikipedia,
  image gen) in the same invoke. A node needing both would need a
  separate research pass first, i.e. two model calls instead of one.
- Provider behavior varies a bit (Gemini vs OpenAI) for nested/optional
  fields — worth a quick smoke test per model, since this notebook swaps
  between `ChatGoogleGenerativeAI` and `ChatOpenAI` in places.
- Locks the node to exactly one schema; less natural if a node might want
  to emit different shapes conditionally.
- **Gotcha hit in practice:** `ChatGoogleGenerativeAI.with_structured_output`
  raises `ValueError: Unsupported schema type <class 'function'>` if the
  object passed in isn't recognized as a Pydantic model, TypedDict, or
  dict (see `langchain_google_genai/chat_models.py`'s `is_typeddict`
  branch). This fired once from a stale kernel state — a leftover
  definition from an earlier edit-and-run cycle was still bound to the
  schema's name. Not a bug in the pattern itself, but a reminder to
  restart-and-rerun-clean when iterating on a schema definition in a long
  -lived notebook kernel.

## 3. Prebuilt ReAct agent + `response_format` (best fix for `researcher_node`)

This directly resolves `with_structured_output`'s biggest con — "can't
interleave tool calls" — without falling back to the manual
`@tool`-as-sink pattern. `langgraph.prebuilt.create_react_agent` (or its
v1 successor, `langchain.agents.create_agent`) runs the tool-calling loop
*and* automatically appends one trailing `with_structured_output`-style
call once the loop finishes — no ReAct loop or closure dict needed:

```python
from langgraph.prebuilt import create_react_agent

class ResearchNotes(TypedDict):
    research: list[str]
    summary: str

research_agent = create_react_agent(
    model=model,
    tools=[wikipedia_tool, tavily_search_tool],
    response_format=ResearchNotes,   # framework makes 1 extra call at the end to shape this
)

def researcher_node(state: BookState) -> dict:
    result = research_agent.invoke({"messages": [("user", f"PROMPT: {state['PROMPT']} ...")]})
    return result["structured_response"]   # already a ResearchNotes dict
```

From the installed source (`chat_agent_executor.py:373-394`): *"The graph
will make a separate call to the LLM to generate the structured response
after the agent loop is finished."* So under the hood it's really "run
tools, then `with_structured_output`" — the same two primitives above,
just composed for you instead of hand-wired via `updates` dict +
`run_agent`. This would let `researcher_node` drop
`make_append_to_state_tool` entirely.

**Pros:** solves the "tools + structured output" combination properly,
one call site, no manual loop.
**Cons:** two full model round trips per node (tool loop, then the
structured pass) instead of one; pulls in the prebuilt-agent state schema
(`messages`-based) instead of this codebase's own `BookState`, so some
glue is needed either way.

### Does `create_react_agent` support a variable number of tool calls?

Yes — this isn't a limitation to work around. Per the source docstring:
*"The process repeats until no more `tool_calls` are present in the
response."* The compiled graph is `agent → tools → agent → tools → ...`,
looping back to the model after every batch of results and exiting only
once the model responds with zero `tool_calls`. Same open-ended shape as
`run_agent`'s manual ReAct loop, not a fixed count. Two details:

- **Per-turn parallel calls:** the `ToolNode` runs "1 tool per
  `tool_call`" — if the model asks for `generate_image` 3 times in one
  turn, all 3 execute before looping back. So it handles both "N calls
  across N turns" and "N calls in one turn."
- **Bounded by `recursion_limit`, not a max-tool-calls parameter.** Every
  LangGraph invocation has a `recursion_limit` (default 25 graph steps);
  each `agent`→`tools` round trip costs 2 of those. Same "generous
  ceiling, not a per-call guarantee" shape as `page_loop`'s
  `max_iterations=40` on the ADK side.

So tool-call-count flexibility was never the reason `@tool`-as-sink fits
`page_node`'s image step better — `create_react_agent` handles "1-3
`generate_image` calls depending on the page" equally well. The actual
reason is the trailing `response_format` synthesis call it adds: that's
exactly what `researcher_node` needs and `page_node`'s image step
doesn't.

**`page_node` rewritten with `create_react_agent`, to show why it doesn't
help there:**

```python
from langgraph.prebuilt import create_react_agent

def page_node(state: BookState) -> dict:
    idx = state["currentPage"]
    outline_entry = state["PAGE_OUTLINE"][idx]

    # Text step: no tools at all, so create_react_agent buys nothing here —
    # a plain model.invoke() already does what run_agent(text_prompt, []) did.
    text_prompt = f"""..."""
    page_text = _extract_text(model.invoke(text_prompt).content)

    # Image step: variable number of generate_image calls, no synthesis needed.
    image_updates: dict = {}
    image_prompt = f"""..."""
    image_agent = create_react_agent(
        model=model,
        tools=[make_generate_image_tool(image_updates, state["book_name"])],
        # no response_format — nothing to synthesize after the tool calls
    )
    image_agent.invoke({"messages": [("user", image_prompt)]})
    # image_updates["image_path"] is populated the same way it always was —
    # the tool's closure still does the capturing, create_react_agent just
    # drives the loop that calls it.

    return {
        "PAGES": [{
            "page_number": outline_entry["page_number"],
            "topic": outline_entry["topic"],
            "text": page_text,
            "image_path": image_updates.get("image_path"),
        }],
        "currentPage": 1,
    }
```

`make_generate_image_tool` is reused **unmodified** — its closure over
`image_updates` still works, since `create_react_agent`'s `ToolNode`
invokes the same `@tool` object `run_agent` did; the output-capture
mechanism doesn't care which loop drives the tool calls. No
`response_format` is passed, since there's no final answer to shape.

Net effect versus the original: **strictly heavier**, not lighter.
`create_react_agent` compiles a small LangGraph graph (`agent` node,
`tools` node, conditional edges, a `recursion_limit`) — real machinery
for something `run_agent`'s five-line manual loop already handled.
Building `image_agent` fresh inside `page_node` on every one of the
~20-25 page iterations means recompiling that graph 20-25 times, work
`run_agent` doesn't incur (no compile step, just a Python `for` loop).
This is the concrete version of the abstract point above:
`create_react_agent(response_format=...)` earns its keep on
`researcher_node`, where the trailing synthesis call is real work being
saved — here, swapping it in only adds graph-compilation overhead for a
capability (variable-count tool calls) `run_agent` already provided for
free.

### Why not `@tool` + `response_format` together, for the image step?

I.e. keep `generate_image` as a tool, but instead of closure-capturing
`image_updates`, have the agent report back the final list of paths via
`response_format=GeneratedImages` (`{"image_paths": list[str]}`). This
looks appealing — it would replace the "side effect captured in a
mutable dict" con from the `@tool`-as-sink section with an official,
validated return value.

The problem: it requires the model to **transcribe** an opaque value
back into its own output, which is a real failure mode. Look at what
`generate_image` actually returns:

```python
def _call_image_model(prompt: str, book_name: str, filename: str) -> str:
    ...
    target_path = os.path.join(f"book_illustrations/{book_name}/", str(uuid.uuid1()) + filename)
    ...
    return target_path

@tool
def generate_image(prompt: str, filename: str) -> dict:
    path = _call_image_model(prompt, book_name, filename)
    updates.setdefault("image_path", []).append(path)
    return {"status": "success", "image_path": path}
```

`target_path` is a UUID-prefixed string like
`25ee8fd6-8643-11f1-878f-25463a4495f3page_01.png` — generated entirely
by Python, never composed by the model. Adding `response_format` would
mean: tool returns the path as a `ToolMessage` → model reads it as text
→ model has to retype that exact 40-character UUID string verbatim into
its final structured answer. LLMs are unreliable at reproducing long
random identifiers byte-for-byte from context — a single dropped
character in the UUID and the reported path points at a file that
doesn't exist. The closure approach has zero transcription risk by
construction, because the literal Python value `_call_image_model`
produced is captured directly — it never passes back through the model.
It also costs an extra full model round trip (the `response_format`
synthesis pass) just to re-report data Python already has with perfect
fidelity.

**The general rule this reveals:** `response_format` is for values that
genuinely require the model's *judgment or synthesis* — `researcher_node`'s
summary is the model reasoning over several search results into prose
nobody else could write. It's the wrong tool for values a tool call
already computed deterministically — those should be captured where
they're created (the closure/`updates` dict), never routed back through
a generation step with no guarantee of verbatim recall.

#### Does dropping the UUID fix it?

Partially — it trades one risk for a different one, it doesn't eliminate
the reason to prefer closure-capture.

**What improves:** without the UUID, the path becomes fully
reconstructible from information the model already has, not an opaque
value handed back to it:

```python
target_path = os.path.join(f"book_illustrations/{book_name}/", filename)
```

`book_name` is already in `state`, and `filename` is the literal string
the model itself chose as a tool-call argument moments earlier (per the
tool's own docstring: `filename: Unique filename to save as, e.g.
page_01_a.png`). Restating it isn't "copy a random string I was just
shown" — it's "repeat something I authored myself." Short-range
self-consistency is far more reliable than verbatim reproduction of an
injected opaque token, so the mistyped-UUID failure mode genuinely goes
away.

**What gets worse:** dropping the UUID reintroduces the exact problem it
exists to prevent — collisions. Re-running `page_node` for the same page
(retry after a bug, regenerating one page) would silently **overwrite**
the earlier image at the same path. Nothing enforces the model's
"unique filename" instruction either — if it ever reuses `page_01_a.png`
within the same page or across a rerun, the second `generate_image` call
clobbers the first with no error. A middle ground that keeps both
properties: a deterministic-but-unique scheme computed by Python rather
than the model transcribing anything, e.g.
`f"{outline_entry['page_number']:02d}_{i}"` (page number + a running
index the tool closure increments itself) instead of `uuid.uuid1()` —
unique by construction, and short enough that transcription risk would
be negligible even if routed through `response_format`.

**What still doesn't go away:** even with short, deterministic,
model-chosen filenames, `response_format` still inserts a generative
step between "what tool calls actually happened" and "what gets
reported" — and generative steps can omit or hallucinate, not just
mistranscribe. If the model calls `generate_image` 3 times but its final
synthesis answer only lists 2 paths, or lists one that doesn't correspond
to any actual call, nothing catches that. The closure
(`updates.setdefault("image_path", []).append(path)`) can't have that
failure mode by construction — it's populated exactly once per real tool
invocation, with no step where the model re-describes what it did. That
structural guarantee is independent of whether the path strings involved
are UUID-heavy or not, and it's the real reason to keep closure-capture
for this node regardless of the filename scheme.

## 4. Forced single tool call (no ReAct loop)

A lighter-weight variant of `@tool`-as-sink: skip `run_agent`'s
multi-turn loop, bind exactly one tool, force the model to call it, and
read the args off one `.invoke()` — no loop, no risk of the model
replying in prose instead of calling the tool (the failure mode that
broke the original `page_node_conversation`):

```python
bound = model.bind_tools([set_outline_tool], tool_choice="set_outline")  # force this exact tool
ai_message = bound.invoke(prompt)
result = ai_message.tool_calls[0]["args"]   # guaranteed present, no ReAct round-trips
```

**Pros:** cheaper than `run_agent`'s loop (one call instead of up to
`max_tool_iterations`), still uses the `@tool`-decorator style this
codebase prefers.
**Cons:** still tool-based rather than schema-based, so it inherits
`with_structured_output`'s single-purpose-per-call limitation without
quite as clean a validation guarantee. Useful middle ground when a node
needs exactly one deterministic tool call and nothing else.

## 5. Manual output parser + prompt instructions

`PydanticOutputParser`/`JsonOutputParser` chained after the model,
relying purely on prompt instructions ("return JSON shaped like...") plus
a parser — no tool, no native JSON mode:

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=PageConversationOutput)
chain = model | parser   # or add parser.get_format_instructions() into the prompt
result = chain.invoke(text_prompt)
```

This is essentially the original (broken) `page_node_conversation`
reimplemented properly — it's what LangChain itself falls back to
internally for TypedDict schemas when a provider lacks native
JSON-schema support (`parser = JsonOutputParser()` in
`with_structured_output`'s own source). Worth knowing it exists as an
explicit, model-agnostic fallback, but it's strictly weaker than
`with_structured_output` since there's no provider-enforced schema — just
a parser that raises if the model didn't comply.

**Pros:** works with any chat model, even ones with no native
structured-output support at all.
**Cons:** no provider-level enforcement — same fragility class as the
original hand-rolled fence-stripping bug this doc's `with_structured_output`
section fixed. Not recommended over `with_structured_output` for any node
already covered above; only relevant if targeting a model/provider
`with_structured_output` doesn't support well.

## When to use which

Ask one question: **does this node need to call any tool other than
"hand back my structured answer"?**

| Node | Needs other tools? | Use |
|---|---|---|
| `outliner_node` | No | `with_structured_output` |
| `cover_node` (title prompt) | No | `with_structured_output` |
| `page_node_conversation` | No | `with_structured_output` |
| `researcher_node` | Yes — Wikipedia, Tavily | `@tool`-as-sink |
| `page_node` (image step) | Yes — `generate_image`, possibly multiple calls | `@tool`-as-sink |
| `emphasis_node` | No external tool, but iterates per-page | Either — good `with_structured_output` candidate |

## Recommendation

Convert nodes whose entire job is "turn this prompt into structured
data" — `outliner_node`, `cover_node`'s title prompt, `page_node_conversation`,
and `emphasis_node` — to `with_structured_output`, since none of them need
interleaved tool use and the ReAct loop is pure overhead for them. Keep
the `@tool`-as-sink pattern for `page_node`'s image step, which needs an
unpredictable number of `generate_image` calls with no final structured
answer to shape. For `researcher_node` specifically, prefer #3
(`create_react_agent`/`create_agent` with `response_format`) over the
current `@tool`-as-sink implementation — it needs both tool calls *and* a
structured final answer, which is exactly the combination #3 solves
without a hand-rolled `updates` dict. Don't do a blanket rewrite — these
patterns solve different problems and this notebook's nodes span more
than one of them.
