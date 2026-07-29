"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createBook } from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const [topic, setTopic] = useState("");
  const [bookName, setBookName] = useState("");
  const [author, setAuthor] = useState("");
  const [numPages, setNumPages] = useState(1);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const { job_id } = await createBook({
        topic,
        book_name: bookName || undefined,
        author: author || "Anonymous",
        num_pages: numPages,
      });
      router.push(`/books/${job_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setSubmitting(false);
    }
  }

  return (
    <main>
      <h1 className="text-3xl font-semibold">Book Composer</h1>
      <p className="mt-2 text-slate-600">
        Describe a book topic and get back an illustrated PDF, comic PDF, and EPUB.
      </p>

      <form onSubmit={handleSubmit} className="mt-8 space-y-6">
        <div>
          <label htmlFor="topic" className="block text-sm font-medium">
            Topic
          </label>
          <textarea
            id="topic"
            required
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            rows={5}
            className="mt-1 w-full rounded-md border border-slate-300 p-2 focus:border-slate-500 focus:outline-none"
            placeholder="Topic: The Hidden Honey Tree: A New Woodland Adventure..."
          />
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="bookName" className="block text-sm font-medium">
              Book name <span className="text-slate-400">(optional)</span>
            </label>
            <input
              id="bookName"
              type="text"
              value={bookName}
              onChange={(e) => setBookName(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 p-2 focus:border-slate-500 focus:outline-none"
              placeholder="woodlandadventure"
            />
          </div>

          <div>
            <label htmlFor="author" className="block text-sm font-medium">
              Author
            </label>
            <input
              id="author"
              type="text"
              value={author}
              onChange={(e) => setAuthor(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 p-2 focus:border-slate-500 focus:outline-none"
              placeholder="Anonymous"
            />
          </div>
        </div>

        <div>
          <label htmlFor="numPages" className="block text-sm font-medium">
            Number of pages
          </label>
          <input
            id="numPages"
            type="number"
            min={1}
            max={20}
            required
            value={numPages}
            onChange={(e) => setNumPages(Number(e.target.value))}
            className="mt-1 w-32 rounded-md border border-slate-300 p-2 focus:border-slate-500 focus:outline-none"
          />
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}

        <button
          type="submit"
          disabled={submitting}
          className="rounded-md bg-slate-900 px-4 py-2 text-white disabled:opacity-50"
        >
          {submitting ? "Starting..." : "Generate book"}
        </button>
      </form>
    </main>
  );
}
