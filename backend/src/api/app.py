"""FastAPI layer wrapping the book-composer LangGraph pipeline.

Reuses the existing graph.invoke(...) pipeline unchanged. Book generation is
slow (LLM + image calls), so requests are handled as background jobs tracked
in an in-memory registry - fine for a single dev process, same posture as the
graph's own in-memory MemorySaver checkpointer.
"""
import json
import os
import re
import threading
import traceback
import uuid
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.graph.main_graph import graph

app = FastAPI(title="Book Composer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
BOOK_OUTPUT_DIR = BACKEND_ROOT / "book_output"
BOOK_ILLUSTRATIONS_DIR = BACKEND_ROOT / "book_illustrations"

BOOK_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")
SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,255}$")

JobStatus = Literal["pending", "running", "done", "error"]

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


class CreateBookRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=4000)
    book_name: Optional[str] = Field(None, max_length=60)
    author: str = Field("Anonymous", max_length=200)
    num_pages: int = Field(1, ge=1, le=20)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:40] or "book"


def _validate_book_name(book_name: str) -> str:
    if not BOOK_NAME_RE.match(book_name):
        raise HTTPException(status_code=400, detail="Invalid book_name")
    return book_name


def _run_generation(job_id: str, book_name: str, topic: str, author: str, num_pages: int) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        graph.invoke(
            {
                "PROMPT": topic,
                "currentPage": 0,
                "num_pages": num_pages,
                "author": author,
                "book_name": book_name,
            },
            config=config,
        )
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
    except Exception as exc:
        traceback.print_exc()
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/books")
def create_book(payload: CreateBookRequest, background_tasks: BackgroundTasks) -> dict:
    base_slug = _slugify(payload.book_name or payload.topic)
    book_name = f"{base_slug}-{uuid.uuid4().hex[:8]}"
    job_id = uuid.uuid4().hex

    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "book_name": book_name, "error": None}

    background_tasks.add_task(
        _run_generation, job_id, book_name, payload.topic, payload.author, payload.num_pages
    )
    return {"job_id": job_id, "book_name": book_name, "status": "pending"}


@app.get("/api/books/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}


def _image_url(book_name: str, image_path: Optional[str]) -> Optional[str]:
    if not image_path:
        return None
    filename = os.path.basename(image_path)
    return f"/api/books/{book_name}/images/{filename}"


@app.get("/api/books/{book_name}")
def get_book_metadata(book_name: str) -> dict:
    _validate_book_name(book_name)
    json_path = BOOK_OUTPUT_DIR / f"{book_name}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Book not found")

    data = json.loads(json_path.read_text())
    title, cover_image = data[0][0], data[0][1]
    pages_data = data[1:]

    summary = ""
    text_path = BOOK_OUTPUT_DIR / f"{book_name}.text"
    if text_path.exists():
        summary = text_path.read_text()

    pages = []
    for page in pages_data:
        image_paths = page.get("image_path") or []
        pages.append(
            {
                "page_number": page.get("page_number"),
                "topic": page.get("topic"),
                "text": page.get("text"),
                "conversation": page.get("conversation"),
                "image_urls": [_image_url(book_name, p) for p in image_paths],
            }
        )
    pages.sort(key=lambda p: p["page_number"])

    return {
        "book_name": book_name,
        "title": title,
        "summary": summary,
        "cover_image_url": _image_url(book_name, cover_image),
        "pages": pages,
    }


@app.get("/api/books/{book_name}/images/{filename}")
def get_book_image(book_name: str, filename: str):
    _validate_book_name(book_name)
    if not SAFE_FILENAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = BOOK_ILLUSTRATIONS_DIR / book_name / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/api/books/{book_name}/download.pdf")
def download_pdf(book_name: str):
    _validate_book_name(book_name)
    path = BOOK_OUTPUT_DIR / f"{book_name}.pdf"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path, media_type="application/pdf", filename=f"{book_name}.pdf")


@app.get("/api/books/{book_name}/download-comic.pdf")
def download_comic_pdf(book_name: str):
    _validate_book_name(book_name)
    path = BOOK_OUTPUT_DIR / f"{book_name}_comic.pdf"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Comic PDF not found")
    return FileResponse(path, media_type="application/pdf", filename=f"{book_name}_comic.pdf")


@app.get("/api/books/{book_name}/download.epub")
def download_epub(book_name: str):
    _validate_book_name(book_name)
    pdf_path = BOOK_OUTPUT_DIR / f"{book_name}.pdf"
    epub_path = BOOK_OUTPUT_DIR / f"{book_name}.epub"
    if not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found; generate the book first")

    if not epub_path.is_file():
        try:
            from src.pdf_writers.pdf_to_kindle_epub import convert_pdf_to_kindle_epub
        except ImportError as exc:
            raise HTTPException(
                status_code=500, detail=f"EPUB conversion unavailable: {exc}"
            ) from exc

        title = book_name
        json_path = BOOK_OUTPUT_DIR / f"{book_name}.json"
        if json_path.exists():
            data = json.loads(json_path.read_text())
            title = data[0][0] or book_name

        try:
            convert_pdf_to_kindle_epub(pdf_path, epub_path, title=title, author=None)
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"EPUB conversion failed: {exc}") from exc

    return FileResponse(
        epub_path, media_type="application/epub+zip", filename=f"{book_name}.epub"
    )
