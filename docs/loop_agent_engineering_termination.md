# ADK `LoopAgent` termination: `PageLoopGuard` vs `exit_loop`, and sizing `max_iterations`

Context: `book_wrtier_illustrated_adk.ipynb`'s `page_loop`, the ADK port of
the LangGraph pipeline's `page_node` + `route_after_page` conditional-edge
loop.

```python
page_loop = LoopAgent(
    name="page_loop",
    sub_agents=[
        PageLoopGuard(name="page_loop_guard"),
        page_writer_agent,
        page_illustrator_agent,
        PageAdvance(name="page_advance"),
    ],
    # Outline targets 20-25 pages; this is a generous safety cap — the loop
    # normally exits via PageLoopGuard's escalate once currentPage catches
    # up to len(PAGE_OUTLINE), same role MAX_ITERATIONS played in the
    # LangGraph movie-pitch example earlier in this project.
    max_iterations=40,
)
```

## How `LoopAgent` executes this list

ADK's `LoopAgent` repeatedly runs its `sub_agents` **in order, top to
bottom**, looping back to the top — until some sub-agent's event sets
`escalate=True`, or `max_iterations` is hit. Escalate stops the loop
**immediately**, mid-pass, without running the remaining sub-agents in
that pass.

The order here is `PageLoopGuard → page_writer_agent → page_illustrator_agent
→ PageAdvance`. Guard is first, so every pass starts by asking "are we
done?" before doing any work.

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

### Walkthrough with a 3-page outline

Say `PAGE_OUTLINE` has 3 entries and state starts at `currentPage=0,
PAGES=[]`.

**Pass 1**
- `PageLoopGuard`: `0 >= 3`? No → `escalate=False`, keep going.
- `page_writer_agent`: writes text for `PAGE_OUTLINE[0]` → `state["_current_page_text"]`
- `page_illustrator_agent`: calls `generate_page_image` → appends to `_current_page_images`
- `PageAdvance`: `idx=0`, appends page 1 to `PAGES`, sets `currentPage=1`, clears image scratch

**Pass 2** — same shape, `idx=1`: writes/illustrates page 2, `PAGES` has 2 entries, `currentPage=2`.

**Pass 3** — `idx=2`: writes/illustrates page 3, `PAGES` has 3 entries, `currentPage=3`.

**Pass 4**
- `PageLoopGuard`: `3 >= 3`? Yes → `escalate=True`.
- Loop stops **right there** — `page_writer_agent`, `page_illustrator_agent`,
  `PageAdvance` never run this pass.

Result: 3 pages written using 4 passes total — the last one being just the
guard check. For the real book (20-25 page outline), this naturally takes
~21-26 passes: N productive passes + 1 trailing guard-only pass to detect
completion.

## Why not `google.adk.tools.exit_loop`?

`exit_loop` is a tool an **LLM agent** calls when it judges its own task
is done — it works by setting `tool_context.actions.escalate = True` from
inside a model's tool-calling turn. That fits judgment calls ("did I
research enough?"), not this loop's exit condition, which is a pure index
comparison: `currentPage >= len(PAGE_OUTLINE)`. There's no reasoning
involved, so `PageLoopGuard`/`EmphasisLoopGuard` encode it as a
deterministic `BaseAgent` instead — the ADK equivalent of the LangGraph
version's plain Python `route_after_page` conditional-edge function, not
an LLM decision.

Using `exit_loop` here would mean attaching it as a tool to
`page_writer_agent` or `page_illustrator_agent` and prompting that LLM to
call it "when you're on the last page," which:

- adds a failure mode — the model can forget to call it (loop runs to
  `max_iterations` every time) or call it early (truncated book)
- couples loop-control logic into a prompt whose actual job is writing
  prose or generating images
- is strictly worse than a one-line list-length check for something that
  has nothing to do with the LLM's task

`exit_loop` earns its keep in loops whose termination genuinely depends on
model judgment (e.g. a research/critique loop deciding "good enough,
stop") — not here.

## Sizing `max_iterations`

`max_iterations` is **not** a per-page allocation or guarantee — it's a
hard ceiling on total passes through the loop, regardless of page count.
In this design each pass processes exactly one page, so the number of
passes needed does track page count, but the relationship only goes one
way: the cap can cut work short, it can't "give" anything.

**Failure mode it guards against:** if `PageAdvance`'s `state_delta`
somehow failed to apply (session-state bug, wrong key) and `currentPage`
never advanced, `PageLoopGuard` would keep seeing `0 >= N` as `False`
forever — the loop would rewrite page 1 over and over, indefinitely,
burning LLM + image-gen calls with no way to stop. `max_iterations` forces
a stop regardless of whether `escalate` ever fires.

**Why 40 and not exactly 25 (the outline's stated max):**

1. **25 would be off by one even in the best case.** A loop with N pages
   needs N productive passes *plus* 1 trailing guard-only pass to detect
   completion and escalate — 26 passes for a 25-page outline, not 25.
2. **26 is the exact minimum, with zero slack.** It only covers the
   *stated* target precisely. Nothing in `outliner_node`/`set_outline`
   enforces the "20-25 pages" instruction — it's a prompt request, not a
   constraint — so a real run could hand back 26+ pages. A cap of exactly
   26 would then truncate a legitimately-generated outline by however
   many pages it overshot, with no error raised.
3. **40 buys headroom for that overshoot** without being so large that a
   genuinely stuck loop (the `currentPage`-never-advances scenario above)
   burns an excessive number of calls before the ceiling kicks in. There's
   no precise derivation for 40 specifically — it's a round number chosen
   to comfortably clear 26 plus a plausible overshoot; 30 or 35 would have
   served just as well.

**Silent-truncation caveat worth remembering:** if the outline ever
exceeds `max_iterations` pages, the `LoopAgent` force-stops without
error — `currentPage` freezes below `len(PAGE_OUTLINE)`, and whatever's
in `PAGES` at that point gets written out as if it were the complete
book. Nothing currently asserts `len(PAGE_OUTLINE) <= max_iterations`.

## Why not fold the whole page loop into one `create_react_agent`/`create_agent` call?

Technically possible — give one agent tools like `write_page_text(page_number,
text)` and `generate_image(...)`, hand it the full outline, tell it "write
all N pages." It would run. But it reintroduces the `exit_loop` problem
above one level up, plus new ones specific to bundling everything into a
single invocation.

**1. Page iteration is deterministic index arithmetic, not a judgment
call — same argument as `PageLoopGuard` vs `exit_loop`, just moved to the
outer loop.** "Process outline entry 0, then 1, ... then N-1, exactly
once each, in order" is `for idx in range(len(PAGE_OUTLINE))`, not
something requiring reasoning. Delegating it to one agentic loop means
trusting the model to self-track "which pages have I already done" purely
from conversation history across dozens of tool calls, with nothing
external checking it — opening the door to skipped pages, duplicated
pages, or the model declaring itself done at page 20 of a 25-page outline
(the same premature-stop failure mode `exit_loop` has, now with no
`PageLoopGuard`-equivalent deterministic check catching it).

**2. Context/cost blows up.** Each page's generation today starts fresh —
only that page's prompt, no history from other pages. One combined
invocation means every prior page's text and every tool call/result stays
in the message list for all subsequent pages: by page 20, the model
re-reads ~19 pages of prior text plus 40-60 tool round trips before even
starting page 20. Token cost and latency grow with book length instead of
staying flat per page.

**3. The safety-cap math gets much harder to size.** Each page needs
~2-4 tool round trips (1 text write + 1-3 images). Across 20-25 pages
that's 40-100+ round trips inside *one* loop, needing a single
`max_iterations` (or LangGraph `recursion_limit`) sized to cover the
entire book — versus the current clean split, where `max_iterations=40`
only bounds *how many pages* and each page's own tool use is separately
and independently bounded.

**4. Retry/debug granularity disappears.** A bad page 14 today means
re-running one identifiable `page_node`/loop pass. One mega-call means a
failure partway through likely means redoing the whole multi-page
invocation, with no clean per-page checkpoint to resume from.

**Net:** `create_react_agent`/`create_agent` can handle a variable number
of tool calls fine (see above) — that was never the blocker. The blocker
is that "iterate through a fixed, known-length list exactly once per
entry" belongs in the graph/loop layer, not delegated to a model's
self-directed judgment about when it's done. Keep the LLM's job scoped to
"write this one page" / "illustrate this one page" — the boundary
`page_node` already draws.

## General rule: when to loop in Python vs let the LLM loop

The litmus test: **can the termination condition be computed from data
already in state, or does it require judging the content of what's
happened so far?** Former → loop in Python. Latter → let the LLM drive it
via tool calls.

### Loop in Python when

- The iteration count is knowable ahead of time from state already held
  (`len(PAGE_OUTLINE)`, `len(PAGES)`).
- Each iteration is an independent, well-scoped unit of work worth
  retrying/inspecting individually.
- Getting the count wrong is a real correctness bug (a missing page isn't
  "close enough").
- Per-iteration cost/context should stay flat instead of accumulating
  across the whole loop.

Examples in this codebase: `page_node`'s outer loop
(`route_after_page`/`PageLoopGuard`), `emphasis_node`'s
`for page in state["PAGES"]`. Both are "process this known-length list,
once per entry, in order" — no reasoning required to know when to stop.

### Let the LLM loop (agentic tool-calling, `run_agent`/`create_react_agent`) when

- The iteration count genuinely isn't knowable in advance — it depends on
  what's discovered mid-loop.
- Continuing or stopping requires evaluating content, not counting — "is
  this enough information," "does this page need 1 image or 3."
- The task is exploratory rather than "walk this known list."

Examples in this codebase: `researcher_node`'s Wikipedia/Tavily loop — how
many searches, and when the research is "enough," is a judgment call
about content, not an index. `page_node`'s image step is the same: "1-3
images depending on whether the text describes a sequence of distinct
moments" requires reading the text, not counting anything. This is also
exactly where `exit_loop`/`escalate`-as-judgment is the *right* tool,
unlike for page iteration — a loop whose real exit condition is "the
model judges the draft/research is good enough" is precisely what that
mechanism is for.

### They nest — `page_node` already does it

`page_node` is the working example of both together: the **outer** loop
(which page) is Python-driven and deterministic; the **inner** loop (how
many `generate_image` calls for *this* page) is LLM-driven and
judgment-based. Don't pick one mode for a whole pipeline — pick per loop.
A Python-controlled outer loop wrapping an LLM-controlled inner loop is
the normal shape, not a special case (the mistake in the previous section
would have been collapsing both levels into one LLM-driven loop, losing
the outer level's determinism for no benefit to the inner level, which
already gets its flexibility either way).

### One rule that applies to both

Even when the LLM legitimately drives the loop, always still cap it in
Python — `run_agent`'s `max_tool_iterations=5`, `page_loop`'s
`max_iterations=40`, LangGraph's `recursion_limit`. The model deciding
*when* to stop doesn't mean nothing should bound *how long you'll wait
for it to decide* — that ceiling is insurance against the loop never
terminating, not a contradiction of "let the LLM handle it." Judgment
picks the normal exit; Python picks the worst-case exit.
