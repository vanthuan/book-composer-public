# Book Composer — frontend

Next.js (App Router) + TypeScript + Tailwind. Talks to the backend API in `../backend`
over plain REST — there's no server-side proxy, the browser calls the backend directly.

## Setup

The system default `node` is too old for Next.js — use nvm:

```bash
nvm use 20        # or 22
npm install
cp .env.local.example .env.local
```

`.env.local` sets `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`) — the backend
must be running and reachable at that URL, with CORS allowing `http://localhost:3000`
(already configured in `backend/src/api/app.py`).

## Run

```bash
npm run dev      # http://localhost:3000
```

Other scripts: `npm run build`, `npm run start` (serve the production build),
`npm run lint`.

## Pages

- `app/page.tsx` — form to submit a topic, optional book name, author, and page count.
  Posts to `POST /api/books` and redirects to the job page.
- `app/books/[jobId]/page.tsx` — polls `GET /api/books/jobs/{jobId}` every 3s. Shows an
  indeterminate "generating" state while pending/running (the backend has no per-step
  progress signal), the error message if the job failed, or — once done — the book's
  title/cover/pages plus download links for PDF, comic PDF, and EPUB.

## API client

`lib/api.ts` has the typed fetch helpers (`createBook`, `getJobStatus`,
`getBookMetadata`, `downloadUrl`, `absoluteImageUrl`) — add new backend calls there
rather than calling `fetch` directly from components.

## Known limitations

- No test suite.
- Generation can take several minutes (LLM + per-page image generation); the status
  page just polls, it doesn't show granular step-by-step progress.
