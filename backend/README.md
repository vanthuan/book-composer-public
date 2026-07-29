# Book Composer — backend

Python 3.12, LangGraph pipeline + a thin FastAPI layer on top of it.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GOOGLE_API_KEY and TAVILY_API_KEY
```

Required env vars (see `.env.example`):

| Var | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini text + image generation |
| `TAVILY_API_KEY` | web search tool used by the researcher node |
| `GEMINI_MODEL` | text model, e.g. `gemini-3.5-flash` |
| `GEMINI_IMAGE_MODEL` | image model, e.g. `gemini-3.1-flash-image` |
| `OPENAI_MODEL` / `OPENAI_IMAGE_MODEL` | unused today (the OpenAI code paths in `src/illustrator/image_model.py` and `src/graph/llm_models.py` are present but not wired into the graph); only needed if you switch the pipeline over to OpenAI |

All commands below assume you're running from this `backend/` directory — output
paths (`book_output/`, `book_illustrations/`) are relative to it.

## Run the API

```bash
uvicorn src.api.app:app --reload --port 8000
```

Endpoints:

- `POST /api/books` — `{topic, book_name?, author?, num_pages}` → starts a background
  generation job, returns `{job_id, book_name, status}`.
- `GET /api/books/jobs/{job_id}` — poll job status (`pending` / `running` / `done` / `error`).
- `GET /api/books/{book_name}` — metadata for a finished book (title, summary, cover,
  per-page text/images) once its job is `done`.
- `GET /api/books/{book_name}/images/{filename}` — serves a generated illustration.
- `GET /api/books/{book_name}/download.pdf` / `/download-comic.pdf` / `/download.epub` —
  the finished artifacts (the EPUB is generated lazily on first request).

Jobs are tracked in an in-memory registry — restarting the server loses in-flight/queued
job status (the underlying files in `book_output/`/`book_illustrations/` are unaffected).

## Run the pipeline directly (no API)

```bash
python main.py
```

Runs `graph.invoke(...)` once with the prompt hardcoded in `main.py` and prints each
page's path when done. Useful for quick manual testing without starting the server.

## Regenerate a PDF from `book_output/`

`pdf_writer_node` saves the full book as `book_output/<book_name>.json` (title, cover
image, and every page dict, including the emphasis phrases picked by the `emphasis`
node) — the PDFs are just a rendering of that JSON, so you can re-render either layout
from it without rerunning the graph (no LLM/API calls involved).

`book_output/` and `book_illustrations/` are gitignored (generated, per-run content),
except for the `lastkid` sample (the one `main.py` generates), kept as a working
example: [`book_output/lastkid.pdf`](book_output/lastkid.pdf),
[`book_output/lastkid_comic.pdf`](book_output/lastkid_comic.pdf), and
[`book_output/lastkid.json`](book_output/lastkid.json) (the source data both are
rendered from).

The comic layout has a ready-made loader for this, `create_comic_book_v2_from_json`:

```python
# from backend/
from src.pdf_writers.create_comic_bookv2 import create_comic_book_v2_from_json

out = create_comic_book_v2_from_json(
    "book_output/lastkid.json",
    output_path="book_output/lastkid_comic_v2.pdf",
    author="Olivertwist",
    kindle=True,
)
print(f"Saved {out}")
```

The standard layout (`create_pdf_book`) has no equivalent loader — build the
`(text, images, emphasis)` pairs it expects directly from the saved JSON:

```python
# from backend/
import json
from src.pdf_writers.create_pdf_book import create_pdf_book

data = json.loads(open("book_output/lastkid.json").read())
title, cover_image = data[0]
pages = data[1:]

pairs = [(p["text"], p["image_path"], p.get("emphasis", [])) for p in pages]

out = create_pdf_book(
    pairs,
    output_path="book_output/lastkid_v2.pdf",
    title=title,
    author="Olivertwist",
    cover_image=cover_image,
    kindle=True,
)
print(f"Saved {out}")
```

This is also how you'd re-render a book after tweaking layout code in
`create_pdf_book.py`/`create_comic_bookv2.py` — no need to burn API credits re-running
the whole pipeline just to see a rendering change.

## Layout

- `src/graph/` — the LangGraph pipeline: `agent_state.py` (state schema), `agents.py`
  (the node functions), `main_graph.py` (graph wiring), `tools.py` (LangChain tools),
  `llm_models.py` (the shared LLM client).
- `src/agent_runner/utils.py` — a small hand-rolled ReAct loop (`run_agent`) that every
  node calls into.
- `src/pdf_writers/` — pure-Python rendering: `create_pdf_book.py` (standard layout),
  `create_comic_bookv2.py` (comic layout), `pdf_to_kindle_epub.py` (PDF → fixed-layout
  EPUB for Kindle).
- `src/illustrator/image_model.py` — Gemini image-generation calls.
- `src/api/app.py` — the FastAPI layer described above.
- `src/skills_runner/` + `skills/` — an experimental alternate illustration path (not
  used by the graph — don't build on it without checking it's still relevant).

## Known limitations

- No tests.
- Jobs run one at a time per process via FastAPI `BackgroundTasks`; there's no queue,
  so many concurrent requests will just run concurrently in threads with no throttling.
- `author` isn't persisted into `book_output/<book_name>.json`, so it isn't shown back
  by `GET /api/books/{book_name}`.
