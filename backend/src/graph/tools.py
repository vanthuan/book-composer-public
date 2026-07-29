from langchain_core.tools import tool
import wikipedia
wikipedia.set_user_agent("book-composer-research-agent/1.0 (contact: acb@gmail.com)")
from langchain_community.tools import WikipediaQueryRun, TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from src.illustrator.image_model import call_image_model

# utility func
def format_list(text_list) -> str:
    return "\n".join(text_list)

# state logger tool
def make_append_to_state_tool(updates: dict):
    @tool
    def append_to_state(field: str, response: str) -> dict:
        """Append new output to an existing state key.

        Args:
            field: The state field name to append to (e.g. 'research').
            response: The string to append to that field.
        """
        updates.setdefault(field, []).append(response)
        return {"status": "success"}

    return append_to_state

## Wikipedia search tool
wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(),
    handle_tool_error=True,
)
## Tavily search tool
tavily_search_tool = TavilySearchResults(max_results=3)

## set outline for the book tool
def make_set_outline_and_characters_tool(updates: dict):
    @tool
    def set_outline_and_characters(
        pages: list[dict],
        characters: dict[str, str],
        summary: str,
    ) -> dict:
        """Set the page-by-page outline, initial cast, and story summary.

        Args:
            pages: List of pages, each a dict with 'page_number' (int) and
                'topic' (str, a one-sentence scene description).
            characters: Appearance description for every named or clearly implied character
                across the outline, keyed by character_name.
            summary: A 2-3 sentence description of the story.
        """
        updates["PAGE_OUTLINE"] = pages
        updates["character_reference"] = characters
        updates["book_summary"] = summary
        return {"status": "success"}

    return set_outline_and_characters

# generate image tool
def make_generate_image_tool(updates: dict, book_name:str):
    @tool
    def generate_image(prompt: str, filename: str, all_actions: int) -> dict:
        """Generate one illustration for the given prompt and save it locally.

        Call this once per distinct image the page needs — most pages need
        just one, but call it again with a new filename for pages that
        describe a sequence of distinct moments best shown as separate images.

        Args:
            prompt: Image generation prompt. Flat illustration style, no text in the image.
            filename: Unique filename to save as, e.g. page_01_a.png, page_01_b.png
        """
        path =  call_image_model(prompt, book_name, filename, all_actions)
        # path = call_openai_image_model(prompt, book_name, filename, all_actions)

        updates.setdefault("image_path", []).append(path)
        return {"status": "success", "image_path": path}

    return generate_image


# set page information tool
def make_set_page_node_coversation_tool(updates: dict):
    @tool
    def set_page_node_converstation(
        story_text: str,
        conversation: list[dict],
        character_updates: dict[str, str],
    ) -> dict:
        """Set the story, conversation, and character appearance notes for the illustrated page

        Args:
            story_text: the text description of the story
            conversation: List of character speech, each a dict with 'character_name' (str) and
                    'speech' (str, a 1-2 sentences of the character dialog)
            character_updates: Appearance description for each character who is new on this
                    page, or whose appearance changes because of what happens in this page's
                    story text. Keyed by character_name. Omit any character whose appearance
                    is unchanged from the existing reference — leave them out entirely so
                    their established look carries forward unmodified.
        """
        updates["story_text"] = story_text
        updates["conversation"] = conversation
        updates["character_updates"] = character_updates
        return {"status": "success"}
    return set_page_node_converstation

# set bold and big phrases for the page story
def make_set_emphasis_tool(updates: dict, page_number: int):
    @tool
    def set_emphasis(phrases: list[dict | str]) -> dict:
        """Set the phrases to visually emphasize in this page's printed text.

        Args:
            phrases: List of phrases to emphasize. Each entry is either a
                plain string (rendered bold, ~1.3x size) or a dict with a
                'text' key and optional 'font_size' (int), 'bold' (bool),
                'italic' (bool) overrides, e.g.
                {"text": "neon-lit skyway", "font_size": 20, "bold": True, "italic": True}.
        """
        updates.setdefault("PAGE_EMPHASIS", {})[page_number] = phrases
        return {"status": "success"}

    return set_emphasis