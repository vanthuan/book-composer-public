import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Book Composer",
  description: "Generate illustrated books and download them as PDF, comic PDF, or EPUB.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900">
        <div className="mx-auto max-w-3xl px-4 py-10">{children}</div>
      </body>
    </html>
  );
}
