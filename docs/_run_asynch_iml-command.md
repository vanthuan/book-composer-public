# ADK's `_run_async_impl` + `Event` vs LangGraph's `Command`

Context: `book_wrtier_illustrated_adk.ipynb`'s custom `BaseAgent` subclasses
(`PageLoopGuard`, `PageAdvance`, `EmphasisLoopGuard`, `EmphasisAdvance`),
and how the same "state update + control flow" problem is solved in the
LangGraph version of this pipeline.

## What `_run_async_impl` is

`_run_async_impl` is the abstract "core logic" hook every
`google.adk.agents.BaseAgent` subclass must override — it's how custom
behavior plugs into ADK's agent execution model. From
`google/adk/agents/base_agent.py`:

```python
async def _run_async_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    """Core logic to run this agent via text-based conversation."""
    raise NotImplementedError(
        f'_run_async_impl for {type(self)} is not implemented.'
    )
    yield  # AsyncGenerator requires having at least one yield statement
```

The base class's default just raises `NotImplementedError` — it exists
purely as a contract subclasses must fulfill.

**How it gets called:** never directly. The public entry point is
`run_async(parent_context)`, which every agent (LLM `Agent`,
`SequentialAgent`, `LoopAgent`, custom `BaseAgent`s) is actually invoked
through. It does framework bookkeeping — builds the `InvocationContext`,
fires `before_agent`/`after_agent` callbacks, wraps everything in
instrumentation — and delegates to the override in the middle:

```python
async def run_async(self, parent_context: InvocationContext) -> AsyncGenerator[Event, None]:
    ctx = self._create_invocation_context(parent_context)
    ...
    async with Aclosing(self._run_async_impl(ctx)) as agen:
        async for event in agen:
            yield event
    ...
```

The underscore prefix signals "this is the part you implement; the public
`run_async` wraps it with plumbing you shouldn't have to reimplement."

**Why an async generator, not a plain async function:** an agent's turn
can produce zero, one, or many `Event`s (text output, state changes, tool
calls). `yield` lets ADK stream events out as they're produced, consumed
one at a time via `async for` by whatever's running the agent
(`LoopAgent`, `SequentialAgent`, the runner).

**How the notebook's deterministic agents use it** — plain Python, one
`Event` yielded:

```python
class PageLoopGuard(BaseAgent):
    """Escalates (stops the surrounding LoopAgent) once every outline page
    has been written."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        done = state.get("currentPage", 0) >= len(state.get("PAGE_OUTLINE", []))
        yield Event(author=self.name, actions=EventActions(escalate=done))


class PageAdvance(BaseAgent):
    """Folds this iteration's scratch state into PAGES, then advances
    currentPage and clears the per-page image scratch list."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        idx = state.get("currentPage", 0)
        outline_entry = state["PAGE_OUTLINE"][idx]
        pages = list(state.get("PAGES", []))
        pages.append({
            "page_number": outline_entry["page_number"],
            "topic": outline_entry["topic"],
            "text": state.get("_current_page_text", ""),
            "image_path": state.get("_current_page_images", []),
        })
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={
                "PAGES": pages,
                "currentPage": idx + 1,
                "_current_page_images": [],
            }),
        )
```

## Why `yield Event(...)` is required, not optional

`Event` is the **only channel** an agent has to communicate two things
outward: "here's a state change" and "here's a control-flow signal." You
can't mutate `ctx.session.state` directly or `return` a value — nothing
downstream would see it. Two concrete reasons, traced through the ADK
source:

**1. It's how `LoopAgent` decides to stop.** From `loop_agent.py`:

```python
async with Aclosing(sub_agent.run_async(ctx)) as agen:
    async for event in agen:
        yield event
        if event.actions.escalate:
            should_exit = True
```

`LoopAgent` doesn't call a special "are you done?" method on
`PageLoopGuard` — it runs it like any sub-agent, consumes whatever
`Event`s come out, and checks `event.actions.escalate` on each one. If
`PageLoopGuard` returned a plain `bool` instead of yielding an `Event`,
there'd be no mechanism for that signal to reach the loop. The `yield`
**is** the API.

**2. It's how state changes actually get committed.** From
`in_memory_session_service.py`'s `append_event`:

```python
if event.actions and event.actions.state_delta:
    ...
    storage_session.state.update(session_state_delta)
```

State only updates when an `Event` carrying `state_delta` is appended to
the session by the runner. `PageAdvance` computing `pages.append(...)`
and `idx + 1` as local Python variables does nothing on its own — those
values persist only because they're packaged into
`EventActions(state_delta={...})` and yielded, so the framework's
event-processing pipeline (`run_async` → runner →
`session_service.append_event`) picks them up and writes them into
`ctx.session.state`.

So `Event` isn't a formality — it's the message format the whole
framework is built around: every agent, whether an LLM `Agent` streaming
tokens or a plain-Python `BaseAgent` like `PageLoopGuard`, reports
everything it does (text, tool calls, state writes, escalation) as a
stream of `Event`s, and that uniform stream is what lets
`SequentialAgent`/`LoopAgent`/the runner observe and react to sub-agents
without needing to know what kind of agent each one is.

## Is this the same as LangGraph's `Command`?

Same conceptual role, different mechanics. Both exist to bundle "a state
update" together with "a control-flow decision" into one signal the
orchestrator consumes — but they're shaped differently.

### The mapping

| ADK `EventActions` | LangGraph `Command` | Role |
|---|---|---|
| `state_delta` | `update` | Merge these values into shared state |
| `escalate` | *(no direct equivalent)* | Loop-scoped: "stop the nearest `LoopAgent`" |
| `transfer_to_agent` | `goto` | Redirect control flow to a named node/agent |

From `langgraph/types.py`:
```python
class Command(Generic[N], ToolOutputMixin):
    """One or more commands to update the graph's state and send messages to nodes."""
    update: Any | None = None
    resume: dict[str, Any] | Any | None = None
    goto: Send | Sequence[Send | N] | N = ()
```

`update` runs through the same reducers as a plain dict return (e.g. this
codebase's `Annotated[list, operator.add]` fields) — it's not a
different state-merge mechanism, just an alternate return shape that also
carries routing. `goto` is the direct analogue of ADK's
`transfer_to_agent`: jump to any named node/agent, not just "stop."

### Where they diverge

**1. Single return vs. a stream.** `Command` is returned once,
synchronously, from a node function — one value, node's done.
`_run_async_impl` is an async generator; an agent can `yield` many
`Event`s over one turn (e.g. an LLM `Agent` streaming several
text-chunk events before a final one). ADK's model is "report everything
as a sequence of small messages"; LangGraph's is "return one final
verdict."

**2. `escalate` has no `goto` equivalent.** `escalate` doesn't say
*where* to go — it's a narrow, binary "unwind out of the nearest
enclosing `LoopAgent`," closer to `break` in a loop than to routing.
LangGraph keeps "what state changed" (node return / `Command.update`)
and "which edge to take" (conditional-edge function) as two separate
mechanisms unless a node deliberately reaches for `Command(goto=...)` to
fuse them. ADK fuses state-delta and loop-exit into the same
`Event`/`EventActions` object unconditionally — there's no separate
"edge function" concept in `LoopAgent`.

**3. Necessity differs.** In LangGraph, returning a plain dict (what
every node in this notebook's `outliner_node`/`page_node`/etc. does) is
enough for state updates — `Command` is opt-in, only needed when a node
also wants to control routing dynamically. In ADK, yielding `Event` isn't
optional in the same way — it's the *only* path state changes or
escalation can travel, as shown by `append_event` being the sole place
`state.update(...)` happens.

## Examples: the same use case, three ways

### 1. What this codebase does today (LangGraph, *not* using `Command`)

State update and routing are two separate mechanisms — a node returns a
plain dict, and a *different* function decides where to go next:

```python
def page_node(state: BookState) -> dict:
    idx = state["currentPage"]
    outline_entry = state["PAGE_OUTLINE"][idx]
    page_text = run_agent(text_prompt, [])
    ...
    return {                      # ← state update only
        "PAGES": [{...}],
        "currentPage": 1,          # reducer adds this, per Annotated[int, operator.add]
    }

def route_after_page(state: BookState) -> str:   # ← routing decided separately
    if state["currentPage"] >= len(state["PAGE_OUTLINE"]):
        return "emphasis"
    return "page_node"

builder.add_conditional_edges("page_node", route_after_page, ["page_node", "emphasis"])
```

### 2. The same thing rewritten with `Command` (fused, like ADK)

```python
from langgraph.types import Command
from typing import Literal

def page_node(state: BookState) -> Command[Literal["page_node", "emphasis"]]:
    idx = state["currentPage"]
    outline_entry = state["PAGE_OUTLINE"][idx]
    page_text = run_agent(text_prompt, [])
    ...
    new_current_page = idx + 1
    done = new_current_page >= len(state["PAGE_OUTLINE"])

    return Command(
        update={"PAGES": [{...}], "currentPage": 1},   # ~ EventActions.state_delta
        goto="emphasis" if done else "page_node",       # ~ EventActions.transfer_to_agent
    )

# No add_conditional_edges call needed at all — page_node routes itself.
builder.add_edge(START, "page_node")   # just a static entry edge now
```

This is the direct structural analogue of ADK's `PageLoopGuard` +
`PageAdvance` — one return value carries both the state delta and the
next-hop decision, instead of splitting them across a node function and a
conditional-edge function.

### 3. ADK's version, side by side

```python
class PageAdvance(BaseAgent):
    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        idx = ctx.session.state.get("currentPage", 0)
        pages = list(ctx.session.state.get("PAGES", []))
        pages.append({...})
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={   # ~ Command.update
                "PAGES": pages,
                "currentPage": idx + 1,
            }),
        )

class PageLoopGuard(BaseAgent):
    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        done = ctx.session.state.get("currentPage", 0) >= len(ctx.session.state.get("PAGE_OUTLINE", []))
        yield Event(actions=EventActions(escalate=done))   # ~ Command's "stop", no goto target needed
```

ADK splits what `Command` could fuse into one call back into *two*
agents (`PageLoopGuard` checks-and-stops, `PageAdvance`
updates-and-continues) — because `LoopAgent`'s sub-agent list is fixed
and ordered, there's no "jump to an arbitrary next agent" the way
`Command(goto=...)` or `transfer_to_agent` allows. `escalate` only ever
means "unwind this loop," so it doesn't need a destination.

### 4. Where `goto`/`transfer_to_agent` actually match — arbitrary redirection

This is the case where the two are closest in spirit: a supervisor
deciding *which* agent handles the next step, not just loop/no-loop.

**LangGraph:**
```python
def supervisor_node(state) -> Command[Literal["researcher", "writer", "__end__"]]:
    next_agent = decide_who_goes_next(state)   # e.g. an LLM classifying intent
    return Command(update={"last_speaker": "supervisor"}, goto=next_agent)
```

**ADK equivalent** (conceptually — an LLM `Agent` whose model decides to
hand off, not something `LoopAgent`/`SequentialAgent` do on their own):
```python
EventActions(transfer_to_agent="writer_agent")
```

Both say the same thing — "state changed, and here's who runs next" —
bundled into one message instead of a separate state-write plus a
separate routing lookup. The `page_node`/`PageAdvance` case in this
notebook doesn't need that generality (it only ever goes to "the same
node again" or "the next fixed step"), which is why the codebase's
current LangGraph version uses the simpler two-mechanism form and ADK's
loop only needs `escalate`, not `transfer_to_agent`.
