from src.graph.agent_state import BookState
from src.agent_runner.utils import run_agent
from src.skills_runner.utils import run_skill
from langchain_google_genai import ChatGoogleGenerativeAI
from src.prompts.all_prompts import (
    CHARACTER_APPEARANCE_RUBRIC,
    ANIME_STYLE_DIRECTIVE
)
from src.graph.tools import (
    wikipedia_tool, tavily_search_tool, make_append_to_state_tool,
    format_list, make_set_outline_and_characters_tool, make_set_page_node_coversation_tool,
    make_set_emphasis_tool, make_generate_image_tool
)
from src.pdf_writers.create_pdf_book import create_pdf_book
from src.pdf_writers.create_comic_bookv2 import create_comic_book_v2
import json
import os
from src.graph.llm_models import model

## Research node — gathers a handful of vivid, concrete story details via Wikipedia/web search before outlining
def researcher_node(state: BookState) -> dict:
    updates: dict = {}
    system_prompt = f"""
    PROMPT:
    {state['PROMPT']}

    INSTRUCTIONS:
    - Use your Wikipedia and web search tools to gather a handful of vivid, concrete
      details about the subject in the PROMPT — the kind of specific images, moments,
      or facts a picture book illustrator and young reader would find memorable.
    - Don't try to be exhaustive — 3-5 good details are enough for a short book.
    - Use the 'append_to_state' tool to add your research to the field 'research'.
    - Summarize what you have learned.
    Now, do your research.
    """
    run_agent(system_prompt, [wikipedia_tool, tavily_search_tool, make_append_to_state_tool(updates)])
    return updates

## Outline node — drafts the page-by-page outline, the initial cast/appearance reference, and a story summary
def outline_node_conversation(state: BookState) -> dict:
    outline_prompt = f"""
    Create a page-by-page outline for a short illustrated story about: {state["PROMPT"]}


    RESEARCH:
    {format_list(state.get('research'))}

    Produce between {state["num_pages"]} and {state["num_pages"] + 1} pages. For each page,
    give a one-sentence topic/scene description that reads like a
    continuous story arc.

    Then list every character who is named or clearly implied across the
    whole outline (main and recurring side characters), and invent a fixed
    appearance for each. This is the reference appearance they must keep
    for the rest of the book.
    
    {CHARACTER_APPEARANCE_RUBRIC}

    Finally write a 2-3 sentence summary of the story.

    Use the 'set_outline_and_characters' tool with your pages, characters,
    and summary.
    """
    updates: dict = {}
    run_agent(outline_prompt, [make_set_outline_and_characters_tool(updates)])
    print(f'####Generated: {len(updates["PAGE_OUTLINE"])} pages.')
    return {
        "PAGE_OUTLINE": updates["PAGE_OUTLINE"],
        "character_reference": updates["character_reference"],
        "book_summary": updates["book_summary"],
    }



## Cover node — writes the book title and generates the cover illustration
def cover_node(state: BookState) -> dict:
    title_prompt = f"""
    PROMPT:
    {state['PROMPT']}

    PAGE_OUTLINE:
    {format_list(str(p) for p in state.get('PAGE_OUTLINE', []))}

    Come up with a short, catchy title for this illustrated picture book.
    Return just the title text, nothing else - no quotes, no explanation.
    """
    book_title = run_agent(title_prompt, []).strip()

    character_reference = state.get("character_reference", {})
    cover_updates: dict = {}
    cover_prompt = f"""
    Design a single striking, eye-caching, professional-look and beautiful cover illustration for a picture book
    titled "{book_title}" and authored by {state["author"]} about: {state['PROMPT']}

    PAGE_OUTLINE:
    {format_list(str(p) for p in state.get('PAGE_OUTLINE', []))}
    
        
    <CHARACTER CONSISTENCY>
    Each character has a fixed appearance established across the book:
    {json.dumps(character_reference, indent=2)}

    Render every character exactly as described above — same skin tone, hair
    color/style, clothes, shoes, and accessories — across this page's entire
    sequence of images and across the whole book. Treat the sequence as one
    continuous scene: a character's appearance must not drift, reset, or vary
    from one image to the next within the sequence, even as their pose, angle,
    or action changes. Do not redesign or vary any part of a character's
    appearance between images unless the PAGE TEXT explicitly describes
    something that changes it partway through the sequence (e.g. changing
    into pajamas, getting rained on, putting on a costume) — in that case,
    keep the new appearance consistent for every image from that point
    onward. If nothing in the page text affects a character's appearance,
    keep it identical to the reference in every image, start to finish.
    </CHARACTER CONSISTENCY>

    {ANIME_STYLE_DIRECTIVE}

    The image should work as a standalone cover: one eye-catching central
    scene capturing the spirit of the story, rich color, flat illustration
    style. 
    
    Do not include any text or lettering in the image itself - the
    title will be printed separately.
    
    Call 'generate_image' once with your prompt and filename="cover.png".
    """
    
    run_agent(cover_prompt, [make_generate_image_tool(cover_updates, state["book_name"])])

    return {
        "BOOK_TITLE": book_title,
        "COVER_IMAGE": (cover_updates.get("image_path") or [None])[0],
    }


## Page node — writes one page's story text and character conversation, then generates its illustration
def page_node_conversation(state: BookState) -> dict:
    idx = state["currentPage"]
    outline_entry = state["PAGE_OUTLINE"][idx]
    page_number = outline_entry['page_number']
    character_reference = state.get("character_reference",{})
    PROMPT = state["PROMPT"]
    

    text_prompt = f"""
    Write the text and conversation for page {page_number} of a short illustrated story about: {PROMPT}

    PAGE TOPIC:
    {outline_entry["topic"]}

    Write 2-4 sentences of story text for this page.
    Basing on the text, infer a conversation for characters in the text. 
    Limit each conversational turn to 1-2 sentences (1 long sentence or 2 short sentences, no more than ~20 words). Limit to at most 1-3 turns.
    Return text and the conversation with characters and their speech.

    <CHARACTER REFERENCE>
    These characters already have a fixed appearance established on earlier pages:
    {json.dumps(character_reference, indent=2) if character_reference else "(none yet — this is the first page)"}

    When you call the tool, use 'character_updates' to record an appearance
    for any character who is new on this page, or whose appearance changes
    because of something that happens in this page's story (e.g. changing
    into pajamas, getting rained on, putting on a costume). Leave any
    character whose look is unchanged out of 'character_updates' entirely —
    their existing reference above carries forward as-is.
    {CHARACTER_APPEARANCE_RUBRIC}
    </CHARACTER REFERENCE>
    """
    updates: dict = {}
    run_agent(text_prompt, [make_set_page_node_coversation_tool(updates)])

    page_text = updates["story_text"]
    conversation = updates["conversation"]
    character_reference.update(updates.get("character_updates") or {})

    '''
    v1
    Read this page's text and conversation, then decide how many illustrations it needs.
    Most pages need just one image, but split into more images (at most 3 images) if the conversation covers distinct speeches from the same characters. 
    If split, order the images to match the conversation and keep each character's appearance consistent across them, since they depict one continuous scene. 
    The images should include all the characters mentioned in the conversation. The characters should be in the same image if they interact directly to each others.

    '''
    image_updates: dict = {}
    image_prompt = f"""
    
    <INSTRUCTION>
    Read this page's text and conversation, then generate a single
    illustration for this page.

    That image should include all story characters' series of actions from the
    page, organized as a left-to-right sequence of moments within the one
    image, each moment separated from the next by a clean vertical bar/divider
    drawn inside the image — like a filmstrip or comic strip laid out across
    one wide frame.

    Order the moments left to right to match the conversation, and keep each
    character's appearance consistent across every moment in the sequence,
    since they depict one continuous scene.

    The image should include all the characters mentioned in the conversation.
    Characters should share the same moment/segment if they interact directly
    with each other at that point in the scene.

    Do not include any speech bubbles, captions, or signage with writing
    in the image — the story text and dialogue are added separately in a
    later step.
    </INSTRUCTION>


    <CHARACTER CONSISTENCY>
    Each character has a fixed appearance established across the book:
    {json.dumps(character_reference, indent=2)}

    Render every character exactly as described above — same skin tone, hair
    color/style, clothes, shoes, and accessories — across this page's entire
    sequence of moments and across the whole book. Treat the sequence as one
    continuous scene: a character's appearance must not drift, reset, or vary
    from one moment to the next within the sequence, even as their pose, angle,
    or action changes. Do not redesign or vary any part of a character's
    appearance between moments unless the PAGE TEXT explicitly describes
    something that changes it partway through the sequence (e.g. changing
    into pajamas, getting rained on, putting on a costume) — in that case,
    keep the new appearance consistent for every moment from that point
    onward. If nothing in the page text affects a character's appearance,
    keep it identical to the reference in every moment, start to finish.
    </CHARACTER CONSISTENCY>

    <CONSTRAINTS>
    {ANIME_STYLE_DIRECTIVE}

    Imagine your image filled over a tall portrait book page. 
    The speech texts in the conversation will be bubbled onto the image in a
    later step — do not draw the bubbles yourself.
    </CONSTRAINTS>

    PAGE TEXT:
    {page_text}

    CONVERSATION:
    {json.dumps(conversation, indent=2)}

    
    Call 'generate_image' exactly once, with your prompt and filename
    page_{page_number:02d}.png. Attach the number of series of actions to all_action 
    """

    run_agent(image_prompt, [make_generate_image_tool(image_updates, state["book_name"])])

    return {
        "PAGES": [{
            "page_number": page_number,
            "topic": outline_entry["topic"],
            "text": page_text,
            "conversation": conversation,
            "image_path": image_updates.get("image_path"),
            "character_reference": character_reference
        }],
        "currentPage": 1,
        "character_reference": character_reference,
    }


## Emphasis node — picks 1-3 short phrases per page worth visually emphasizing in the printed book
def emphasis_node(state: BookState) -> dict:
    updates: dict = {"PAGE_EMPHASIS": {}}
    for page in state["PAGES"]:
        system_prompt = f"""
        PAGE TEXT:
        {page['text']}

        INSTRUCTIONS:
        Pick 1-3 short phrases from the PAGE TEXT above that are worth visually
        emphasizing in the printed book — vivid imagery, key actions, or
        important names. Keep each phrase short (a few words).

        Use the 'set_emphasis' tool with your list of phrases. If nothing in
        the text is worth emphasizing, call it with an empty list.
        """
        run_agent(system_prompt, [make_set_emphasis_tool(updates, page["page_number"])])
    return updates


## PDF writer node — renders the finished book (standard + comic-style) PDFs and saves the book JSON/summary
def pdf_writer_node(state: BookState) -> dict:
    emphasis_by_page = state.get("PAGE_EMPHASIS", {})
    pairs = [
        (
            page["text"],
            page["image_path"],
            emphasis_by_page.get(page["page_number"], [])
        )
        for page in sorted(state["PAGES"], key=lambda p: p["page_number"])
    ]
    pages = state["PAGES"]
    for page in pages:
        page.update({"emphasis":emphasis_by_page.get(page["page_number"], [])})

    os.makedirs("book_output", exist_ok=True)
    
    with open(f'book_output/{state["book_name"]}.json', "w") as fp:
        json.dump([(state.get("BOOK_TITLE") or state.get("PROMPT"), state.get("COVER_IMAGE"))]+ pages, fp, indent=2)
    with open(f'book_output/{state["book_name"]}.text', "w") as fp:
        fp.write(state["book_summary"])

    output_path = create_pdf_book(
            pairs,
            output_path=os.path.join("book_output", f"{state["book_name"]}.pdf"),
            title=state.get("BOOK_TITLE") or state.get("PROMPT"),
            author=state["author"],
            cover_image=state.get("COVER_IMAGE"),
            kindle=True
            #layout="scrapbook"
        )
    print(f"Wrote {output_path}")

    output_comic_path = create_comic_book_v2(
                pages,
                output_path=os.path.join("book_output", f"{state["book_name"]}_comic.pdf"),
                title=state.get("BOOK_TITLE") or state.get("PROMPT"),
                author=state["author"],
                cover_image=state.get("COVER_IMAGE"),
                kindle=True
                #layout="scrapbook"
            )
    print(f"Wrote {output_comic_path}")
    return {}

## Router (not an agent node) — loops back to page_node_conversation until every outline page is written, then continues to emphasis
def route_after_page(state: BookState) -> str:
    if state["currentPage"] >= len(state["PAGE_OUTLINE"]):
        return "emphasis"
    return "page_node_conversation"


### Uncomment to use skill if you want
'''
def page_node_conversation(state: BookState) -> dict:
    idx = state["currentPage"]
    outline_entry = state["PAGE_OUTLINE"][idx]
    page_number = outline_entry['page_number']
    character_reference = state.get("character_reference", {})
    PROMPT = state["PROMPT"]

    text_prompt = f"""
    Write the text and conversation for page {page_number} of a short illustrated story about: {PROMPT}

    PAGE TOPIC:
    {outline_entry["topic"]}

    Write 2-4 sentences of story text for this page.
    Basing on the text, infer a conversation for characters in the text. 
    Limit each conversational turn to 1-2 sentences. Limit to at most 3 turns.
    Return text and the conversation with characters and their speech.

    <CHARACTER REFERENCE>
    These characters already have a fixed appearance established on earlier pages:
    {json.dumps(character_reference, indent=2) if character_reference else "(none yet — this is the first page)"}

    When you call the tool, use 'character_updates' to record an appearance
    for any character who is new on this page, or whose appearance changes
    because of something that happens in this page's story (e.g. changing
    into pajamas, getting rained on, putting on a costume). Leave any
    character whose look is unchanged out of 'character_updates' entirely —
    their existing reference above carries forward as-is.
    {CHARACTER_APPEARANCE_RUBRIC}
    </CHARACTER REFERENCE>
    """
    updates: dict = {}
    run_agent(text_prompt, [make_set_page_node_coversation_tool(updates)])

    page_text = updates["story_text"]
    conversation = updates["conversation"]
    character_reference.update(updates.get("character_updates") or {})

    image_updates = run_skill(
        "page-illustrator",
        model= model,
        book_name=state["book_name"],
        task=f"""
        PAGE TEXT:
        {page_text}

        CONVERSATION:
        {json.dumps(conversation, indent=2)}

        CHARACTER REFERENCE:
        {json.dumps(character_reference, indent=2)}
        """,
    )

    return {
        "PAGES": [{
            "page_number": page_number,
            "topic": outline_entry["topic"],
            "text": page_text,
            "conversation": conversation,
            "image_path": image_updates.get("image_path"),
            "character_reference": character_reference
        }],
        "currentPage": 1,
        "character_reference": character_reference,
    }
'''