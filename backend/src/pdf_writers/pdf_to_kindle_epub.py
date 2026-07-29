"""Convert a laid-out PDF (e.g. from create_pdf_book.py) into a fixed-layout
EPUB that Amazon KDP accepts for Kindle picture books.

Amazon KDP does not accept a raw PDF upload for a Kindle ebook (only for a
paperback's *print interior*), and there is no Kindle file format literally
called "KDF". Amazon's own path for image-heavy picture books is either the
Kindle Create GUI tool (which outputs a .kpf) or a fixed-layout EPUB, which
KDP's converter turns into the Kindle format (KFX) automatically on
publish. Since create_pdf_book.py already lays out each page exactly as
intended, this renders each PDF page to a PNG and wraps it as one
fixed-layout EPUB page — the same image-per-page approach Kindle Kids' Book
Creator uses for picture books — so the Kindle edition matches the PDF
pixel-for-pixel instead of risking reflowed text/images.

Requires: pip install pymupdf
"""

from __future__ import annotations

import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import fitz  # pymupdf

NAV_XHTML = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><meta charset="utf-8"/><title>Navigation</title></head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Contents</h1>
    <ol>
      <li><a href="pages/page_001.xhtml">Start</a></li>
    </ol>
  </nav>
</body>
</html>
"""

STYLES_CSS = """html, body { margin: 0; padding: 0; }
.page { width: 100%; height: 100%; }
img { display: block; width: 100%; height: 100%; object-fit: contain; }
"""

CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def _render_pages(pdf_path: Path, dpi: int) -> list[tuple[bytes, int, int]]:
    """Render each PDF page to PNG bytes. Returns [(png_bytes, width_px, height_px), ...]."""
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        pages.append((pix.tobytes("png"), pix.width, pix.height))
    doc.close()
    return pages


def _page_xhtml(image_filename: str, width: int, height: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width={width}, height={height}"/>
  <link rel="stylesheet" type="text/css" href="../styles.css"/>
</head>
<body>
  <div class="page">
    <img src="../images/{image_filename}" alt=""/>
  </div>
</body>
</html>
"""


def _content_opf(
    title: str,
    author: str | None,
    pages: list[tuple[str, str, int, int]],
    book_id: str,
) -> str:
    modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    creator = f'<dc:creator>{_xml_escape(author)}</dc:creator>' if author else ""

    manifest_items = [
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '<item id="css" href="styles.css" media-type="text/css"/>',
    ]
    spine_items = []
    for i, (image_filename, page_filename, width, height) in enumerate(pages, start=1):
        manifest_items.append(
            f'<item id="img{i:03d}" href="images/{image_filename}" media-type="image/png"/>'
        )
        manifest_items.append(
            f'<item id="page{i:03d}" href="pages/{page_filename}" '
            f'media-type="application/xhtml+xml"/>'
        )
        spine_items.append(f'<itemref idref="page{i:03d}"/>')

    manifest = "\n    ".join(manifest_items)
    spine = "\n    ".join(spine_items)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="BookId"
         prefix="rendition: http://www.idpf.org/vocab/rendition/#">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="BookId">urn:uuid:{book_id}</dc:identifier>
    <dc:title>{_xml_escape(title)}</dc:title>
    {creator}
    <dc:language>en</dc:language>
    <meta property="dcterms:modified">{modified}</meta>
    <meta property="rendition:layout">pre-paginated</meta>
    <meta property="rendition:orientation">portrait</meta>
    <meta property="rendition:spread">none</meta>
  </metadata>
  <manifest>
    {manifest}
  </manifest>
  <spine>
    {spine}
  </spine>
</package>
"""


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def convert_pdf_to_kindle_epub(
    pdf_path: str | Path,
    output_path: str | Path,
    title: str,
    author: str | None = None,
    dpi: int = 300,
) -> Path:
    """Render `pdf_path` page-by-page into a fixed-layout EPUB at `output_path`,
    ready to upload directly to KDP (KDP converts it to Kindle format on publish).

    Args:
        pdf_path: Source PDF, e.g. one built by create_pdf_book.py.
        output_path: Where to write the .epub.
        title: Book title (EPUB metadata).
        author: Optional author name (EPUB metadata).
        dpi: Render resolution for each page image. 300 is print-quality;
            drop to ~200 for a smaller file if upload size is a concern.

    Returns:
        Path to the written .epub.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    book_id = str(uuid.uuid4())

    rendered = _render_pages(pdf_path, dpi=dpi)
    pages: list[tuple[str, str, int, int]] = []
    for i, (_, width, height) in enumerate(rendered, start=1):
        pages.append((f"page_{i:03d}.png", f"page_{i:03d}.xhtml", width, height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as epub:
        # mimetype must be the first entry, stored (uncompressed), per the
        # EPUB OCF spec — a compressed or non-first mimetype makes some
        # readers/validators (including KDP's) reject the file outright.
        epub.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", zipfile.ZIP_STORED)

        epub.writestr("META-INF/container.xml", CONTAINER_XML, zipfile.ZIP_DEFLATED)
        epub.writestr("OEBPS/styles.css", STYLES_CSS, zipfile.ZIP_DEFLATED)
        epub.writestr("OEBPS/nav.xhtml", NAV_XHTML, zipfile.ZIP_DEFLATED)
        epub.writestr(
            "OEBPS/content.opf",
            _content_opf(title, author, pages, book_id),
            zipfile.ZIP_DEFLATED,
        )

        for (png_bytes, _, _), (image_filename, page_filename, width, height) in zip(rendered, pages):
            epub.writestr(f"OEBPS/images/{image_filename}", png_bytes, zipfile.ZIP_DEFLATED)
            epub.writestr(
                f"OEBPS/pages/{page_filename}",
                _page_xhtml(image_filename, width, height),
                zipfile.ZIP_DEFLATED,
            )

    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python pdf_to_kindle_epub.py <input.pdf> <title> [author]")
        sys.exit(1)

    pdf_arg, title_arg = sys.argv[1], sys.argv[2]
    author_arg = sys.argv[3] if len(sys.argv) > 3 else None
    out = convert_pdf_to_kindle_epub(
        pdf_arg,
        Path(pdf_arg).with_suffix(".epub"),
        title=title_arg,
        author=author_arg,
    )
    print(f"Wrote {out}")
