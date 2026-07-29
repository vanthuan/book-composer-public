"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { use as usePromise } from "react";
import {
  getJobStatus,
  getBookMetadata,
  absoluteImageUrl,
  downloadUrl,
  type JobStatusResponse,
  type BookMetadata,
} from "@/lib/api";

const POLL_INTERVAL_MS = 3000;

export default function BookJobPage({ params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = usePromise(params);
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [book, setBook] = useState<BookMetadata | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        const status = await getJobStatus(jobId);
        if (cancelled) return;
        setJob(status);

        if (status.status === "done") {
          const metadata = await getBookMetadata(status.book_name);
          if (!cancelled) setBook(metadata);
        } else if (status.status === "pending" || status.status === "running") {
          timer = setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (err) {
        if (!cancelled) setPollError(err instanceof Error ? err.message : "Failed to fetch status");
      }
    }

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [jobId]);

  if (pollError) {
    return (
      <main>
        <p className="text-sm text-red-600">{pollError}</p>
        <Link href="/" className="mt-4 inline-block text-slate-600 underline">
          Back to form
        </Link>
      </main>
    );
  }

  if (!job || job.status === "pending" || job.status === "running") {
    return (
      <main>
        <h1 className="text-2xl font-semibold">Generating your book...</h1>
        <p className="mt-2 text-slate-600">
          This runs research, outline, cover, and per-page illustration steps and can take several
          minutes. This page updates automatically.
        </p>
      </main>
    );
  }

  if (job.status === "error") {
    return (
      <main>
        <h1 className="text-2xl font-semibold text-red-700">Generation failed</h1>
        <p className="mt-2 whitespace-pre-wrap text-sm text-red-600">{job.error}</p>
        <Link href="/" className="mt-4 inline-block text-slate-600 underline">
          Try again
        </Link>
      </main>
    );
  }

  if (!book) {
    return (
      <main>
        <p className="text-slate-600">Loading book...</p>
      </main>
    );
  }

  return (
    <main>
      <Link href="/" className="text-sm text-slate-500 underline">
        &larr; New book
      </Link>
      <h1 className="mt-2 text-3xl font-semibold">{book.title}</h1>

      {book.cover_image_url && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={absoluteImageUrl(book.cover_image_url) ?? undefined}
          alt={`Cover for ${book.title}`}
          className="mt-4 max-w-sm rounded-md border border-slate-200"
        />
      )}

      <p className="mt-4 text-slate-700">{book.summary}</p>

      <div className="mt-6 flex flex-wrap gap-3">
        <a
          href={downloadUrl(book.book_name, "pdf")}
          className="rounded-md bg-slate-900 px-4 py-2 text-white"
        >
          Download PDF
        </a>
        <a
          href={downloadUrl(book.book_name, "comic")}
          className="rounded-md border border-slate-300 px-4 py-2"
        >
          Download comic PDF
        </a>
        <a
          href={downloadUrl(book.book_name, "epub")}
          className="rounded-md border border-slate-300 px-4 py-2"
        >
          Download EPUB
        </a>
      </div>

      <h2 className="mt-10 text-xl font-semibold">Pages</h2>
      <div className="mt-4 space-y-6">
        {book.pages.map((page) => (
          <div key={page.page_number} className="rounded-md border border-slate-200 p-4">
            <p className="text-sm font-medium text-slate-500">Page {page.page_number}</p>
            {page.image_urls[0] && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={absoluteImageUrl(page.image_urls[0]) ?? undefined}
                alt={page.topic}
                className="mt-2 max-w-md rounded-md border border-slate-200"
              />
            )}
            <p className="mt-2 text-slate-700">{page.text}</p>
          </div>
        ))}
      </div>
    </main>
  );
}
