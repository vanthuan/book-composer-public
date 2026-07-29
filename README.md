# Book Composer

Generates illustrated children's books from a short topic prompt, using a LangGraph
multi-agent pipeline, and lets you download the result as a PDF, a comic-style PDF,
or a Kindle-ready EPUB.

## How it works

A LangGraph pipeline (`backend/src/graph/`) runs a fixed sequence of steps:

```
researcher -> outliner -> cover -> page (loop, one iteration per page) -> emphasis -> pdf_writer
```

- **researcher** gathers a few concrete facts about the topic (Wikipedia + Tavily search).
- **outliner** turns those into a page-by-page outline plus a character reference sheet
  (fixed appearance descriptions, reused on every page so illustrations stay consistent).
- **cover** writes a title and generates a cover illustration.
- **page** (looped once per outline entry) writes the page text, character dialogue, and
  one composited illustration per page.
- **emphasis** picks phrases on each page to bold/enlarge for print.
- **pdf_writer** renders the finished book to `book_output/<book_name>.pdf` (and a
  comic-style variant), plus a JSON/text dump of the book.

See `docs/graph.png` for the rendered graph, and `docs/*.md` for design-rationale notes
on why the pipeline is shaped this way (structured-output pattern, loop termination, etc.).

## Project layout

- `backend/` — the LangGraph pipeline plus a FastAPI layer that runs it as a job and
  serves the generated files. See `backend/README.md`.
- `frontend/` — a Next.js app to submit a book topic and download the result once it's
  ready. See `frontend/README.md`.
- `docs/` — architecture/design-decision notes, not end-user documentation.

## Running it

You need both servers running at once:

```bash
# terminal 1
cd backend
uvicorn src.api.app:app --reload --port 8000

# terminal 2
cd frontend
nvm use 20   # the system default node is too old for Next.js
npm install
cp .env.local.example .env.local
npm run dev
```

Then open http://localhost:3000, submit a topic, and wait — generation involves several
LLM calls plus one image-generation call per page, so it can take a few minutes and
costs real API credits (Gemini + Tavily).

For manual/one-off testing without the web UI, `backend/main.py` runs the pipeline once
with a hardcoded prompt: `cd backend && python main.py`.
