const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type JobStatus = "pending" | "running" | "done" | "error";

export interface CreateBookPayload {
  topic: string;
  book_name?: string;
  author: string;
  num_pages: number;
}

export interface CreateBookResponse {
  job_id: string;
  book_name: string;
  status: JobStatus;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  book_name: string;
  error: string | null;
}

export interface BookPage {
  page_number: number;
  topic: string;
  text: string;
  conversation: { character_name: string; speech: string }[];
  image_urls: (string | null)[];
}

export interface BookMetadata {
  book_name: string;
  title: string;
  summary: string;
  cover_image_url: string | null;
  pages: BookPage[];
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail ? JSON.stringify(body.detail) : res.statusText;
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

export function createBook(payload: CreateBookPayload): Promise<CreateBookResponse> {
  return requestJson("/api/books", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return requestJson(`/api/books/jobs/${jobId}`);
}

export function getBookMetadata(bookName: string): Promise<BookMetadata> {
  return requestJson(`/api/books/${bookName}`);
}

export function absoluteImageUrl(path: string | null): string | null {
  if (!path) return null;
  return `${API_URL}${path}`;
}

export function downloadUrl(bookName: string, kind: "pdf" | "comic" | "epub"): string {
  const suffix = kind === "pdf" ? "download.pdf" : kind === "comic" ? "download-comic.pdf" : "download.epub";
  return `${API_URL}/api/books/${bookName}/${suffix}`;
}
