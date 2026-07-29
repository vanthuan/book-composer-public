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
