"""Build a comic-style PDF straight from the book-writer agent's saved page
data (the {"page_number", "image_path", "conversation", ...} dicts
pdf_writer_node writes to book_output/*.json) — like create_commic_book.py,
but rendering dialogue as plain credited lines instead of drawn speech
bubbles:

    Leo: "Let's make sure we measure the flour exactly, Maya."
    Maya: "I've got the measuring cup ready!"

Reuses create_pdf_book.py's image-enlarging, same-page-preferred sandwich
layout (text blocks interleaved around images, each image sized off the
actual leftover space) and cover-page treatment rather than
re-implementing them — the only real difference from create_pdf_book.py
itself is what goes in each text block: a group of conversation turns
(character name bolded, speech quoted, one per line) instead of a slice of
a single flowing caption.

Requires: pip install reportlab pillow
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer
from reportlab.platypus.flowables import _listWrapOn

# Same-directory import that works whether this module is loaded as
# `src.create_comic_bookv2` (e.g. `from src.create_comic_bookv2 import ...`
# from a notebook with the book-composer root on sys.path) or run/imported
# standalone (`python src/create_comic_bookv2.py`) — see create_comic_book.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_pdf_book import (  # noqa: E402
    CONTENT_WIDTH,
    KINDLE_FONT_SCALE,
    KINDLE_IMAGE_SCALE,
    MARGIN,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    ImageInput,
    _AuthorFlowable,
    _fitted_rl_image,
    _TitleFlowable,
)

Turn = dict  # {"character_name": str, "speech": str}

# create_pdf_book.py's FIT_SAFETY_MARGIN (18pt) isn't enough here: measured
# against real book_output/*.json conversation data, pages that measured
# as fitting with 24-67pt to spare still overflowed in the real render —
# this module's turn-blocks are each a single Paragraph with manual <br/>
# breaks between turns rather than separate Paragraphs, and that renders
# with more real height than _listWrapOn's measurement predicts. Padded
# well past the worst observed gap rather than re-guessing a tighter one.
V2_FIT_SAFETY_MARGIN = 70


def _distribute_turns(conversation: list[Turn], n_groups: int) -> list[list[Turn]]:
    """Split `conversation` into `n_groups` groups as evenly as possible,
    earlier groups getting any remainder — mirrors create_pdf_book.py's
    _split_text_into_blocks, but turns stay intact (never split mid-line)
    rather than being broken at sentence boundaries.
    """
    n = len(conversation)
    base, extra = divmod(n, n_groups)
    groups: list[list[Turn]] = []
    idx = 0
    for i in range(n_groups):
        size = base + (1 if i < extra else 0)
        groups.append(conversation[idx:idx + size])
        idx += size
    return groups


def _build_conversation_markup(turns: list[Turn]) -> str:
    """Build reportlab Paragraph markup for a group of conversation turns,
    one per line: "**Name:** "speech"" — bold name, quoted speech.
    """
    lines = []
    for turn in turns:
        name = _xml_escape(turn["character_name"])
        speech = _xml_escape(turn["speech"])
        lines.append(f'<b>{name}:</b> “{speech}”')
    return "<br/>".join(lines)


def create_comic_book_v2(
    pages_data: list[dict],
    output_path: str | Path = "comic_book_v2.pdf",
    title: str | None = None,
    author: str | None = None,
    cover_image: ImageInput | None = None,
    kindle: bool = False,
) -> Path:
    """Render book-writer agent page dicts into a PDF with each page's
    conversation split into credited text blocks sandwiched around its
    image(s) — block0, image0, block1, image1, ..., imageN-1, blockN —
    the same layout create_pdf_book.py's linear layout uses for split
    captions, just with turn-formatted text instead of sentence-split
    prose.

    Args:
        pages_data: List of page dicts, each with at least "page_number",
            "image_path" (a path or list of paths), and "conversation" (a
            list of {"character_name", "speech"} dicts).
        output_path: Where to write the PDF.
        title: Optional title rendered on a cover page.
        author: Optional author name rendered below the title.
        cover_image: Optional cover illustration. No cover page is added
            unless at least one of `title`, `author`, or `cover_image` is
            given.
        kindle: Scale text/images up for PDFs headed into
            pdf_to_kindle_epub.py (see create_pdf_book.py for why).

    Returns:
        Path to the written PDF.
    """
    output_path = Path(output_path)
    font_scale = KINDLE_FONT_SCALE if kindle else 1.0
    image_scale = KINDLE_IMAGE_SCALE if kindle else 1.0

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "Conversation",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=12 * font_scale,
        leading=16 * font_scale,
        spaceAfter=16 * font_scale,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=(PAGE_WIDTH, PAGE_HEIGHT),
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )
    story: list = []

    if title or author or cover_image:
        if cover_image:
            cover_height = (PAGE_HEIGHT - 2 * MARGIN) * 0.65
            story.append(_fitted_rl_image(cover_image, CONTENT_WIDTH, cover_height))
            story.append(Spacer(1, 0.5 * inch))
        else:
            story.append(Spacer(1, 2 * inch))
        if title:
            story.append(_TitleFlowable(title, base_font_size=36 * font_scale))
        if author:
            story.append(_AuthorFlowable(author, font_size=15 * font_scale))
        story.append(PageBreak())

    _measure_canvas = Canvas(io.BytesIO())
    ordered_pages = sorted(pages_data, key=lambda p: p["page_number"])

    for i, page_data in enumerate(ordered_pages):
        images = page_data["image_path"]
        if isinstance(images, str):
            images = [images]
        n = len(images)
        if n == 0:
            continue

        conversation = page_data.get("conversation") or []
        spacer_height = 0.2 * inch
        available_height = PAGE_HEIGHT - 2 * MARGIN - 12

        turn_groups = _distribute_turns(conversation, n + 1)
        block_paragraphs = [
            Paragraph(_build_conversation_markup(turns), body_style) for turns in turn_groups
        ]
        block_text_heights = [
            _listWrapOn([p], CONTENT_WIDTH, _measure_canvas)[1] for p in block_paragraphs
        ]

        def _build_flow(image_budgets: list) -> list:
            flow = [block_paragraphs[0]]
            for image, budget, block_p in zip(images, image_budgets, block_paragraphs[1:]):
                flow.append(_fitted_rl_image(image, CONTENT_WIDTH, budget))
                flow.append(Spacer(1, spacer_height))
                flow.append(block_p)
            return flow

        # Same "ideal budget -> same-page-preferred proportional scale ->
        # spread-to-separate-pages fallback" algorithm as
        # create_pdf_book.py's linear layout — see that module for the
        # full reasoning. Each image's ideal budget is what it'd get alone
        # on a full page next to just its own leading+trailing block.
        ideal_budgets = [
            max(available_height - block_text_heights[j] - block_text_heights[j + 1], 0) * image_scale
            for j in range(n)
        ]
        leftover_for_images = available_height - spacer_height * n - sum(block_text_heights)
        total_ideal = sum(ideal_budgets)
        fits_together = leftover_for_images > 0 and total_ideal > 0

        if fits_together:
            scale = min(leftover_for_images / total_ideal, 1.0)
            image_budgets = [b * scale for b in ideal_budgets]
            for _ in range(8):
                pair_flowables = _build_flow(image_budgets)
                _, measured_height = _listWrapOn(pair_flowables, CONTENT_WIDTH, _measure_canvas)
                if measured_height <= available_height - V2_FIT_SAFETY_MARGIN:
                    break
                image_budgets = [b * 0.85 for b in image_budgets]
            flowable_groups = [pair_flowables]
        else:
            flowable_groups = []
            for j in range(n):
                image_budget = ideal_budgets[j]
                block: list = []
                for _ in range(8):
                    block = [
                        block_paragraphs[j],
                        _fitted_rl_image(images[j], CONTENT_WIDTH, image_budget),
                        Spacer(1, spacer_height),
                    ]
                    _, block_measured_height = _listWrapOn(block, CONTENT_WIDTH, _measure_canvas)
                    if block_measured_height <= available_height - V2_FIT_SAFETY_MARGIN:
                        break
                    image_budget *= 0.85
                flowable_groups.append(block)
            if turn_groups[n]:
                flowable_groups.append([block_paragraphs[n]])

        for group in flowable_groups:
            story.append(KeepTogether(group))
        if i < len(ordered_pages) - 1:
            story.append(PageBreak())

    doc.build(story)
    return output_path


def create_comic_book_v2_from_json(
    json_path: str | Path,
    output_path: str | Path = "comic_book_v2.pdf",
    author: str | None = None,
    kindle: bool = False,
) -> Path:
    """Load a book_output/*.json file (as saved by pdf_writer_node — the
    first entry is [title, cover_image], the rest are page dicts) and
    render it with credited conversation lines instead of speech bubbles.
    """
    data = json.loads(Path(json_path).read_text())
    title, cover_image = data[0]
    return create_comic_book_v2(
        data[1:],
        output_path=output_path,
        title=title,
        author=author,
        cover_image=cover_image,
        kindle=kindle,
    )


if __name__ == "__main__":
    out = create_comic_book_v2_from_json(
        "book_output/bakeit_2.5pv3.json",
        output_path="book_output/bakeit_2.5pv3_comic_v2.pdf",
        author="Olivertwist",
        kindle=True,
    )
    print(f"Saved {out}")
