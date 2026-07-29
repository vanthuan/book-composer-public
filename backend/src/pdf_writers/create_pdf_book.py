"""Build a PDF book from (text_description, images) pairs.

Each pair becomes one page: the text description, followed by its
associated image or image sequence. Images may be file paths, PIL
Image objects, or raw bytes (matching what generate_image() and the
Gemini image-generation cells in agent_writers.ipynb already produce).

Two layouts are supported:
  - "linear": the caption is split into one more block than there are
    images and interleaved as block0, image0, block1, image1, ...,
    imageN-1, blockN — each image sandwiched between a leading and
    trailing slice of text — with every image enlarged to use as much of
    the leftover page space as it can, rather than one full paragraph
    sitting above a uniformly-sized image/image sequence.
  - "scrapbook": each page is randomly partitioned into varied-size regions
    for the text and each image (treemap-style split), with a little random
    rotation/scale jitter per image for a collage feel.

Emphasis: words/phrases to emphasize can be specified two ways:
  1. Inline markdown in text_description: "**bold phrase**" / "*italic phrase*".
  2. Separately, via an optional third pair element - a list of words/
     phrases to find and emphasize within the text. Each entry can be a
     plain string (bold, ~1.3x size, the default look) or a dict giving
     per-word control over font_size/bold/italic, e.g.:
        ("A tech CEO stands atop a neon-lit skyway at sunset.", image,
         [
             {"text": "neon-lit skyway", "font_size": 20, "bold": True, "italic": True},
             "sunset",  # plain string -> bold, default size bump
         ])
     Matching is case-insensitive; longer phrases take priority over
     substrings. When a separate emphasize list is given for a pair, it
     takes precedence over any inline markdown in that pair's text.

Requires: pip install reportlab
"""

from __future__ import annotations

import io
import random
import re
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from PIL import Image as PILImage
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    Frame,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)
from reportlab.platypus.flowables import _listWrapOn

ImageInput = str | Path | bytes | PILImage.Image
Images = ImageInput | list[ImageInput]
EmphasisEntry = str | dict
Pair = tuple[str, Images] | tuple[str, Images, list[EmphasisEntry]]
Rect = tuple[float, float, float, float]  # x, y, width, height

PAGE_WIDTH, PAGE_HEIGHT = LETTER
MARGIN = 0.75 * inch
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

# The shrink-to-fit loops below check a KeepTogether group's height against
# available_height using the same _listWrapOn measurer KeepTogether itself
# uses — but KeepTogether re-measures with the real document canvas and the
# frame's actual remaining height at render time, not this module's scratch
# canvas. A borderline pass (measured just barely <= available_height) can
# still overflow in the real render from small discrepancies between the
# two, spilling a couple of stray characters onto their own near-blank
# page. This buffer makes the loops keep shrinking until there's genuine
# headroom instead of stopping right at the edge.
FIT_SAFETY_MARGIN = 18

# Base-14 fonts: always available in reportlab, no font-registration needed.
SCRAPBOOK_FONTS = [
    "Helvetica",
    "Helvetica-Oblique",
    "Times-Roman",
    "Times-Italic",
    "Courier",
    "Courier-Oblique",
]

# Bold/italic/bold-italic variants for each base font, used to render
# emphasized words in whichever style each emphasis entry asks for.
FONT_VARIANTS = {
    "Helvetica": {
        "bold": "Helvetica-Bold", "italic": "Helvetica-Oblique",
        "bold_italic": "Helvetica-BoldOblique",
    },
    "Helvetica-Oblique": {
        "bold": "Helvetica-BoldOblique", "italic": "Helvetica-Oblique",
        "bold_italic": "Helvetica-BoldOblique",
    },
    "Times-Roman": {
        "bold": "Times-Bold", "italic": "Times-Italic",
        "bold_italic": "Times-BoldItalic",
    },
    "Times-Italic": {
        "bold": "Times-BoldItalic", "italic": "Times-Italic",
        "bold_italic": "Times-BoldItalic",
    },
    "Courier": {
        "bold": "Courier-Bold", "italic": "Courier-Oblique",
        "bold_italic": "Courier-BoldOblique",
    },
    "Courier-Oblique": {
        "bold": "Courier-BoldOblique", "italic": "Courier-Oblique",
        "bold_italic": "Courier-BoldOblique",
    },
}

_BOLD_EMPHASIS_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_EMPHASIS_RE = re.compile(r"\*(.+?)\*")


def _apply_emphasis_markup(text: str, font_name: str, font_size: float) -> str:
    """Convert inline **bold** / *italic* markers into reportlab font tags.

    `**phrase**` is rendered in the bold variant of `font_name` at a larger
    size; `*phrase*` is rendered in the italic variant at the base size.
    The rest of the text is escaped so raw '&'/'<'/'>' don't break
    reportlab's paragraph markup parser.
    """
    variants = FONT_VARIANTS.get(font_name, FONT_VARIANTS["Helvetica"])
    emphasis_size = font_size * 1.3

    escaped = _xml_escape(text)
    escaped = _BOLD_EMPHASIS_RE.sub(
        lambda m: f'<font name="{variants["bold"]}" size="{emphasis_size:.1f}">{m.group(1)}</font>',
        escaped,
    )
    escaped = _ITALIC_EMPHASIS_RE.sub(
        lambda m: f'<font name="{variants["italic"]}">{m.group(1)}</font>',
        escaped,
    )
    return escaped


def _resolve_emphasis_entry(
    entry: EmphasisEntry, base_font: str, base_size: float
) -> tuple[str, str, float]:
    """Resolve one emphasize entry into (phrase, font_name, font_size).

    A plain string defaults to bold at ~1.3x the base size (the original
    behavior). A dict can override any of "font_size", "bold", "italic":
        {"text": "neon-lit skyway", "font_size": 20, "bold": True, "italic": True}
    """
    if isinstance(entry, str):
        phrase, font_size, bold, italic = entry, None, True, False
    else:
        phrase = entry["text"]
        font_size = entry.get("font_size")
        bold = entry.get("bold", True)
        italic = entry.get("italic", False)

    variants = FONT_VARIANTS.get(base_font, FONT_VARIANTS["Helvetica"])
    if bold and italic:
        font_name = variants["bold_italic"]
    elif bold:
        font_name = variants["bold"]
    elif italic:
        font_name = variants["italic"]
    else:
        font_name = base_font

    resolved_size = font_size if font_size is not None else (base_size * 1.3 if bold else base_size)
    return phrase, font_name, resolved_size


def _apply_emphasis_words(
    text: str, emphasize: list[EmphasisEntry], font_name: str, font_size: float
) -> str:
    """Wrap each occurrence of the given words/phrases in their own font/size.

    Matching is case-insensitive against the plain `text`. Longer phrases
    are tried before shorter ones so a listed phrase takes priority over a
    listed word that happens to be its substring. Text outside a match is
    escaped so raw '&'/'<'/'>' don't break reportlab's markup parser.
    """
    if not emphasize:
        return _xml_escape(text)

    specs = [_resolve_emphasis_entry(entry, font_name, font_size) for entry in emphasize]
    order = sorted(range(len(specs)), key=lambda i: len(specs[i][0]), reverse=True)
    pattern = re.compile(
        "|".join(f"(?P<e{i}>{re.escape(specs[i][0])})" for i in order),
        re.IGNORECASE,
    )

    parts = []
    last_end = 0
    for match in pattern.finditer(text):
        idx = int(match.lastgroup[1:])
        _, em_font, em_size = specs[idx]
        parts.append(_xml_escape(text[last_end:match.start()]))
        parts.append(
            f'<font name="{em_font}" size="{em_size:.1f}">'
            f'{_xml_escape(match.group(0))}</font>'
        )
        last_end = match.end()
    parts.append(_xml_escape(text[last_end:]))
    return "".join(parts)


def _build_emphasis_markup(
    text: str,
    font_name: str,
    font_size: float,
    emphasize: list[EmphasisEntry] | None = None,
) -> str:
    """Emphasize `text` using an explicit word list if given, else inline markdown."""
    if emphasize:
        return _apply_emphasis_words(text, emphasize, font_name, font_size)
    return _apply_emphasis_markup(text, font_name, font_size)


def _unpack_pair(pair: Pair) -> tuple[str, Images, list[EmphasisEntry] | None]:
    """Split a pair into (text, images, emphasize), defaulting emphasize to None."""
    if len(pair) == 3:
        return pair
    text_description, images = pair
    return text_description, images, None


def _split_text_into_blocks(text: str, n_blocks: int) -> list[str]:
    """Split `text` into `n_blocks` chunks along sentence ('.') boundaries.

    Sentences are distributed as evenly as possible (earlier blocks get the
    extra ones when it doesn't divide evenly). Used for multi-image pairs so
    each image gets its own slice of the caption to sit beside instead of
    the whole caption sitting above every image. If there are fewer
    sentences than blocks, the trailing blocks come back empty.
    """
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    base, extra = divmod(len(sentences), n_blocks)
    blocks = []
    idx = 0
    for i in range(n_blocks):
        size = base + (1 if i < extra else 0)
        blocks.append(" ".join(sentences[idx:idx + size]))
        idx += size
    return blocks


def _as_pil_image(image: ImageInput) -> PILImage.Image:
    """Normalize a path/bytes/PIL image into a PIL Image."""
    if isinstance(image, PILImage.Image):
        return image
    if isinstance(image, bytes):
        return PILImage.open(io.BytesIO(image))
    return PILImage.open(image)


def _fitted_rl_image(image: ImageInput, max_width: float, max_height: float) -> RLImage:
    """Build a reportlab Image scaled to fit within max_width x max_height."""
    pil_image = _as_pil_image(image)
    width, height = pil_image.size
    # reportlab's Image treats a width/height of exactly 0 as "unset" and
    # falls back to the source's native pixel size instead of rendering at
    # zero size — clamp so an exhausted height budget yields a genuinely
    # tiny image instead of a full-size one blowing out the layout.
    scale = max(min(max_width / width, max_height / height, 1.0), 1e-6)

    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)
    return RLImage(buffer, width=width * scale, height=height * scale)


def _split_rect(rect: Rect, n: int, rng: random.Random) -> list[Rect]:
    """Recursively partition `rect` into `n` sub-rects of varied size.

    Splits along the longer axis at a random fraction each time, so leaf
    rects end up with organically varied sizes rather than a uniform grid.
    """
    if n <= 1:
        return [rect]

    x, y, w, h = rect
    frac = rng.uniform(0.3, 0.7)
    if w >= h:
        w1 = w * frac
        a, b = (x, y, w1, h), (x + w1, y, w - w1, h)
    else:
        h1 = h * frac
        a, b = (x, y + h - h1, w, h1), (x, y, w, h - h1)

    n_a = n // 2
    n_b = n - n_a
    return _split_rect(a, n_a, rng) + _split_rect(b, n_b, rng)


def _draw_image_in_rect(
    c: Canvas, image: ImageInput, rect: Rect, rng: random.Random
) -> None:
    """Draw `image` inside `rect`, contain-fit, with a light random jitter."""
    pil_image = _as_pil_image(image).convert("RGB")
    x, y, w, h = rect

    # Shrink the slot by a random margin, then contain-fit the image inside it.
    margin_frac = rng.uniform(0.05, 0.15)
    slot_w, slot_h = w * (1 - margin_frac), h * (1 - margin_frac)
    img_w, img_h = pil_image.size
    scale = min(slot_w / img_w, slot_h / img_h) * rng.uniform(0.85, 1.0)
    draw_w, draw_h = img_w * scale, img_h * scale

    cx, cy = x + w / 2, y + h / 2
    angle = rng.uniform(-6, 6)

    c.saveState()
    c.translate(cx, cy)
    c.rotate(angle)
    c.setFillColorRGB(1, 1, 1)
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    border = 6
    c.rect(-draw_w / 2 - border, -draw_h / 2 - border, draw_w + 2 * border, draw_h + 2 * border, fill=1, stroke=1)
    c.drawImage(
        ImageReader(pil_image),
        -draw_w / 2,
        -draw_h / 2,
        width=draw_w,
        height=draw_h,
    )
    c.restoreState()


TITLE_COLOR = (0.13, 0.15, 0.35)      # deep indigo
TITLE_SHADOW_COLOR = (0, 0, 0)
TITLE_RULE_COLOR = (0.80, 0.62, 0.13)  # warm gold accent
TITLE_FONT = "Times-Bold"


def _draw_fancy_title(
    c: Canvas,
    text: str,
    cx: float,
    cy: float,
    max_width: float,
    base_font_size: float = 36,
    font_name: str = TITLE_FONT,
) -> float:
    """Draw a cover-style title: bold serif, soft drop shadow, a thin gold
    rule above/below, auto-shrunk to fit max_width. Returns the font size
    actually used, so callers can size surrounding layout around it.
    """
    font_size = base_font_size
    while font_size > 14 and c.stringWidth(text, font_name, font_size) > max_width:
        font_size -= 1

    rule_width = min(max_width, c.stringWidth(text, font_name, font_size) + 0.6 * inch)
    # Asymmetric gaps: a bold serif's cap-height sits ~0.7em above the
    # baseline, so a symmetric ~0.55em gap put the top rule right through
    # the tops of tall letters. Top gets more clearance than bottom.
    rule_gap_top = font_size * 0.85
    rule_gap_bottom = font_size * 0.5

    c.saveState()
    c.setStrokeColorRGB(*TITLE_RULE_COLOR)
    c.setLineWidth(1.2)
    c.line(cx - rule_width / 2, cy + rule_gap_top, cx + rule_width / 2, cy + rule_gap_top)
    c.line(cx - rule_width / 2, cy - rule_gap_bottom, cx + rule_width / 2, cy - rule_gap_bottom)

    c.setFont(font_name, font_size)
    shadow_offset = max(font_size * 0.03, 1)
    c.setFillColorRGB(*TITLE_SHADOW_COLOR)
    c.drawCentredString(cx + shadow_offset, cy - shadow_offset, text)
    c.setFillColorRGB(*TITLE_COLOR)
    c.drawCentredString(cx, cy, text)
    c.restoreState()
    return font_size


class _TitleFlowable(Flowable):
    """Platypus Flowable wrapper around `_draw_fancy_title`, for use in a
    `SimpleDocTemplate` story where the cover title needs the same shadowed,
    gold-ruled treatment as the scrapbook layout's raw-Canvas cover.
    """

    def __init__(self, text: str, width: float = CONTENT_WIDTH, base_font_size: float = 36):
        super().__init__()
        self.text = text
        self.width = width
        self.base_font_size = base_font_size
        self.height = base_font_size * 2.1

    def draw(self) -> None:
        _draw_fancy_title(
            self.canv,
            self.text,
            self.width / 2,
            self.height * 0.42,
            max_width=self.width,
            base_font_size=self.base_font_size,
        )


AUTHOR_COLOR = (0.45, 0.33, 0.13)  # muted bronze, echoes the title's gold rule
AUTHOR_FONT = "Times-Italic"


def _draw_fancy_author(c: Canvas, author: str, cx: float, cy: float, font_size: float = 15) -> None:
    """Draw the author line as soft italic bronze, flanked by em-dashes,
    instead of the previous plain black 'by {author}' line."""
    text = f"— {author} —"
    c.saveState()
    c.setFont(AUTHOR_FONT, font_size)
    c.setFillColorRGB(*AUTHOR_COLOR)
    c.drawCentredString(cx, cy, text)
    c.restoreState()


class _AuthorFlowable(Flowable):
    """Platypus Flowable wrapper around `_draw_fancy_author`, matching
    `_TitleFlowable`'s pattern so the linear layout gets the same treatment
    as the scrapbook layout's raw-Canvas cover.
    """

    def __init__(self, author: str, width: float = CONTENT_WIDTH, font_size: float = 15):
        super().__init__()
        self.author = author
        self.width = width
        self.font_size = font_size
        self.height = font_size * 1.8

    def draw(self) -> None:
        _draw_fancy_author(self.canv, self.author, self.width / 2, self.height * 0.35, font_size=self.font_size)


def _random_text_style(rng: random.Random, font_scale: float = 1.0) -> ParagraphStyle:
    """Build a paragraph style with a randomized font family and size."""
    font_name = rng.choice(SCRAPBOOK_FONTS)
    font_size = rng.uniform(10, 17) * font_scale
    alignment = rng.choice([0, 1])  # 0=left, 1=center
    return ParagraphStyle(
        "ScrapbookText",
        fontName=font_name,
        fontSize=font_size,
        leading=font_size * 1.3,
        alignment=alignment,
    )


def _render_scrapbook_page(
    c: Canvas,
    text_description: str,
    images: list[ImageInput],
    rng: random.Random,
    emphasize: list[EmphasisEntry] | None = None,
    font_scale: float = 1.0,
) -> None:
    """Randomly partition one page into a text region and image regions."""
    content_box: Rect = (MARGIN, MARGIN, CONTENT_WIDTH, PAGE_HEIGHT - 2 * MARGIN)
    x, y, w, h = content_box

    side = rng.choice(["top", "bottom", "left", "right"])
    text_frac = rng.uniform(0.2, 0.45)

    if side == "top":
        text_rect = (x, y + h * (1 - text_frac), w, h * text_frac)
        images_rect = (x, y, w, h * (1 - text_frac))
    elif side == "bottom":
        text_rect = (x, y, w, h * text_frac)
        images_rect = (x, y + h * text_frac, w, h * (1 - text_frac))
    elif side == "left":
        text_rect = (x, y, w * text_frac, h)
        images_rect = (x + w * text_frac, y, w * (1 - text_frac), h)
    else:  # right
        text_rect = (x + w * (1 - text_frac), y, w * text_frac, h)
        images_rect = (x, y, w * (1 - text_frac), h)

    # Jitter the text block's position/size a little within its slot, and
    # rotate it slightly, for a handwritten-caption feel.
    tx, ty, tw, th = text_rect
    pad_frac = rng.uniform(0.0, 0.08)
    tx, ty = tx + tw * pad_frac / 2, ty + th * pad_frac / 2
    tw, th = tw * (1 - pad_frac), th * (1 - pad_frac)
    angle = rng.uniform(-3, 3)
    text_style = _random_text_style(rng, font_scale)
    marked_up_text = _build_emphasis_markup(
        text_description, text_style.fontName, text_style.fontSize, emphasize
    )

    c.saveState()
    cx, cy = tx + tw / 2, ty + th / 2
    c.translate(cx, cy)
    c.rotate(angle)
    frame = Frame(
        -tw / 2, -th / 2, tw, th,
        leftPadding=6, rightPadding=6, topPadding=6, bottomPadding=6,
        showBoundary=0,
    )
    frame.addFromList([Paragraph(marked_up_text, text_style)], c)
    c.restoreState()

    for rect in _split_rect(images_rect, len(images), rng):
        _draw_image_in_rect(c, images.pop(0), rect, rng)


KINDLE_FONT_SCALE = 1.5  # body 12pt -> 18pt
KINDLE_IMAGE_SCALE = 1.25  # bias the starting image budget up too — otherwise
                            # bigger kindle text (which always keeps its full
                            # required height) squeezes images smaller, not
                            # bigger, since images are what flexes to fit.


def create_pdf_book(
    pairs: list[Pair],
    output_path: str | Path = "book.pdf",
    title: str | None = None,
    author: str | None = None,
    cover_image: ImageInput | None = None,
    layout: str = "linear",
    random_seed: int | None = None,
    kindle: bool = False,
) -> Path:
    """Render (text_description, images) pairs into a paginated PDF book.

    Args:
        pairs: List of (text_description, images) tuples. `images` can be
            a single image (path/bytes/PIL.Image) or a list of images,
            which are laid out as a sequence beneath the text.
        output_path: Where to write the PDF.
        title: Optional title rendered on a cover page.
        author: Optional author name rendered below the title on the cover page.
        cover_image: Optional image rendered on the cover page above the
            title (contain-fit, centered). No cover page is added unless at
            least one of `title`, `author`, or `cover_image` is given.
        layout: "linear" (text above uniformly-sized, stacked images) or
            "scrapbook" (randomly partitioned regions with varied image
            sizes, light rotation, and randomized text placement per page).
        random_seed: Seed for the "scrapbook" layout's randomization, for
            reproducible output.
        kindle: Scale body/title/author text up by `KINDLE_FONT_SCALE`
            (~1.65x). Meant for PDFs headed into pdf_to_kindle_epub.py,
            where each page gets flattened to an image and shrunk to fit a
            6-7in screen — unlike a normal reflowable Kindle ebook, a
            fixed-layout page's font size can't be adjusted by the reader
            afterward, so text sized for a full Letter page read up close
            ends up too small once it's fixed-layout on a small screen.

    Returns:
        Path to the written PDF.
    """
    font_scale = KINDLE_FONT_SCALE if kindle else 1.0

    output_path = Path(output_path)
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["BodyText"],
        fontSize=12 * font_scale,
        leading=16 * font_scale,
        spaceAfter=16 * font_scale,
    )

    if layout == "scrapbook":
        rng = random.Random(random_seed)
        c = Canvas(str(output_path), pagesize=LETTER)

        if title or author or cover_image:
            title_block_height = 1.2 * inch if (title or author) else 0
            if cover_image:
                pil_image = _as_pil_image(cover_image).convert("RGB")
                img_w, img_h = pil_image.size
                max_w = CONTENT_WIDTH
                max_h = PAGE_HEIGHT - 2 * MARGIN - title_block_height
                scale = min(max_w / img_w, max_h / img_h, 1.0)
                draw_w, draw_h = img_w * scale, img_h * scale
                img_y = MARGIN + title_block_height + (max_h - draw_h) / 2
                c.drawImage(
                    ImageReader(pil_image),
                    (PAGE_WIDTH - draw_w) / 2,
                    img_y,
                    width=draw_w,
                    height=draw_h,
                    mask="auto",
                )
            if title:
                _draw_fancy_title(
                    c, title, PAGE_WIDTH / 2, MARGIN + 0.65 * inch,
                    max_width=CONTENT_WIDTH, base_font_size=36 * font_scale,
                )
            if author:
                _draw_fancy_author(c, author, PAGE_WIDTH / 2, MARGIN + 0.3 * inch, font_size=15 * font_scale)
            c.showPage()

        for pair in pairs:
            text_description, images, emphasize = _unpack_pair(pair)
            image_list = list(images if isinstance(images, list) else [images])
            _render_scrapbook_page(c, text_description, image_list, rng, emphasize, font_scale=font_scale)
            c.showPage()

        c.save()
        return output_path

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )

    story = []

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

    # Scratch canvas used only for _listWrapOn's font-metrics access below —
    # never written to a file.
    _measure_canvas = Canvas(io.BytesIO())

    for i, pair in enumerate(pairs):
        text_description, images, emphasize = _unpack_pair(pair)
        image_list = images if isinstance(images, list) else [images]
        n = len(image_list)
        spacer_height = 0.2 * inch
        # SimpleDocTemplate's default Frame adds its own topPadding/
        # bottomPadding (6pt each, reportlab/platypus/frames.py) on top of
        # leftMargin/rightMargin/topMargin/bottomMargin — 12pt of usable
        # height this budget was missing, which was just enough for pairs
        # sitting close to the boundary to measure as "fits" here but not
        # in the real frame.
        available_height = PAGE_HEIGHT - 2 * MARGIN - 12
        image_scale = KINDLE_IMAGE_SCALE if kindle else 1.0

        # Predicting a block's total height by hand (paragraph.wrap() +
        # arithmetic for the images) was a guess about what reportlab would
        # actually measure — spaceBefore/spaceAfter, rounding, and Frame
        # padding all diverge from a hand-rolled formula. Any gap was
        # sometimes just enough to trip KeepTogether's own fit check (see
        # its wrap()/split() in reportlab/platypus/flowables.py — it always
        # forces a split() call, which measures independently via
        # _listWrapOn) and produce a stray blank page. So measure with that
        # same function instead of predicting, and shrink images — never
        # the text — until the real measured height actually fits.
        if n == 0:
            marked_up_text = _build_emphasis_markup(
                text_description, body_style.fontName, body_style.fontSize, emphasize
            )
            flowable_groups = [[Paragraph(marked_up_text, body_style)]]
        else:
            # A single full-width paragraph stacked above every image caps
            # each image's height at whatever's left after the *whole*
            # caption, regardless of how much room the rest of the page
            # has. Splitting the caption into n+1 blocks and sandwiching
            # each image between a leading and trailing block instead
            # — block0, image0, block1, image1, ..., imageN-1, blockN, all
            # still at full page width — means every image only competes
            # with its own two (usually short) neighboring blocks for
            # height, so it can be enlarged to use nearly all of what's
            # left over rather than being capped by one large block up
            # front.
            text_blocks = _split_text_into_blocks(text_description, n + 1)
            block_paragraphs = [
                Paragraph(
                    _build_emphasis_markup(block_text, body_style.fontName, body_style.fontSize, emphasize),
                    body_style,
                )
                for block_text in text_blocks
            ]
            block_text_heights = [
                _listWrapOn([p], CONTENT_WIDTH, _measure_canvas)[1] for p in block_paragraphs
            ]

            def _build_flow(image_budgets: list) -> list:
                # A Spacer only goes after each *image* — not after every
                # text block too — matching the single-block original: two
                # stacked Paragraphs need no explicit gap between them, but
                # an Image has no built-in trailing space of its own.
                flow = [block_paragraphs[0]]
                for image, budget, block_p in zip(image_list, image_budgets, block_paragraphs[1:]):
                    flow.append(_fitted_rl_image(image, CONTENT_WIDTH, budget))
                    flow.append(Spacer(1, spacer_height))
                    flow.append(block_p)
                return flow

            # Each image's *ideal* budget is what it would get alone on a
            # full page, alongside just its own leading and trailing block
            # (ignoring every other image/block on the page). Prefer
            # keeping everything on the same page: scale all the ideal
            # budgets down together by the same factor so they collectively
            # fit in the actual leftover space, which preserves their
            # relative proportions — an image flanked by shorter text still
            # ends up bigger than one flanked by longer text, rather than
            # every image being forced to the same size.
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
                    if measured_height <= available_height - FIT_SAFETY_MARGIN:
                        break
                    image_budgets = [b * 0.85 for b in image_budgets]
                flowable_groups = [pair_flowables]
            else:
                # The blocks alone already fill (or nearly fill) one page —
                # forcing everything onto that single page would starve
                # every image to nothing. Fall back to one
                # leading-block-then-image group per image (each
                # independently sized off its own full page and kept
                # together so it can move to its own page as needed), plus
                # the final trailing block as its own group at the end —
                # the same graceful multi-page spread the multi-image
                # layout already falls back to, just without the (now
                # unaffordable) trailing block sandwiched onto each image.
                flowable_groups = []
                for j in range(n):
                    image_budget = ideal_budgets[j]
                    block: list = []
                    for _ in range(8):
                        block = [
                            block_paragraphs[j],
                            _fitted_rl_image(image_list[j], CONTENT_WIDTH, image_budget),
                            Spacer(1, spacer_height),
                        ]
                        _, block_measured_height = _listWrapOn(block, CONTENT_WIDTH, _measure_canvas)
                        if block_measured_height <= available_height - FIT_SAFETY_MARGIN:
                            break
                        image_budget *= 0.85
                    flowable_groups.append(block)
                flowable_groups.append([block_paragraphs[n]])

        for group in flowable_groups:
            story.append(KeepTogether(group))

        if i < len(pairs) - 1:
            story.append(PageBreak())

    doc.build(story)
    return output_path


if __name__ == "__main__":
    demo_pairs: list[Pair] = [
        (
            "A futuristic city with flying cars at sunset, cinematic lighting.",
            "generated-images/output_img.jpg",
        ),
    ]
    create_pdf_book(demo_pairs, output_path="book.pdf", title="Demo Book")
    create_pdf_book(
        demo_pairs, output_path="book_scrapbook.pdf", title="Demo Book", layout="scrapbook"
    )
    print("Saved book.pdf and book_scrapbook.pdf")
