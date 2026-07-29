from typing import TypedDict, Annotated
import operator

class BookPage(TypedDict):
    page_number: int
    topic: str
    text: str
    conversation: dict
    image_path: list[str] | None
    character_reference: dict | None

class BookState(TypedDict):
    PROMPT: str
    research: Annotated[list[str], operator.add]
    PAGE_OUTLINE: Annotated[list[dict], operator.add]
    PAGES: Annotated[list[BookPage], operator.add]
    PAGE_EMPHASIS: dict[int, list]  # page_number -> emphasize list for create_pdf_book
    BOOK_TITLE: str
    COVER_IMAGE: str | None
    currentPage: Annotated[int, operator.add]
    book_name: str
    author: str
    book_summary: str
    num_pages: int
    character_reference: dict | None
