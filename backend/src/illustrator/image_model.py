import os
import uuid

from google import genai
from google.genai.types import GenerateContentConfig, ImageConfig
import base64
from openai import OpenAI



def call_image_model(prompt: str, book_name: str, filename: str, all_actions: int) -> str:
    """Generate an illustration via the Gemini image model and save it to disk.

    Args:
        prompt: Image generation prompt.
        book_name: Book identifier — the image is saved under
            book_illustrations/<book_name>/.
        filename: Base filename to save as; a UUID is prepended to keep
            filenames unique across calls.
        all_actions: Number of left-to-right moments/actions composed into
            the image. Drives the requested aspect ratio: 1 -> "3:4",
            2 -> "3:2", 3+ -> "21:9".

    Returns:
        Path to the saved image file.
    """
    image_client = genai.Client()
    response = image_client.models.generate_content(
        model=os.getenv("GEMINI_IMAGE_MODEL"),
        contents=prompt,
        config=GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=ImageConfig(
                aspect_ratio="3:4" if all_actions == 1 else "3:2" if all_actions == 2 else "21:9"
            ),
            candidate_count=1,
        ),
    )
    candidates = response.candidates or []
    if not candidates or not candidates[0].content.parts:
        raise RuntimeError(
            f"Image generation returned no image data for prompt (book={book_name}, file={filename}); "
            f"likely blocked by safety filters or model returned text instead of an image."
        )
    image_bytes = candidates[0].content.parts[0].inline_data.data
    target_path = os.path.join(f"book_illustrations/{book_name}/", str(uuid.uuid1()) + filename)
    os.makedirs(f"book_illustrations/{book_name}", exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(image_bytes)
    return target_path

def call_openai_image_model(prompt: str, book_name: str, filename: str, all_actions: int) -> str:
    """Generate an illustration via the OpenAI image model and save it to disk.

    Args:
        prompt: Image generation prompt.
        book_name: Book identifier — the image is saved under
            book_illustrations/<book_name>/.
        filename: Base filename to save as; a UUID is prepended to keep
            filenames unique across calls.
        all_actions: Number of left-to-right moments/actions composed into
            the image. Drives the requested pixel size (OpenAI has no
            aspect-ratio param): 1 -> "1024x1360" (~3:4), 2 -> "1536x1024"
            (~3:2), 3+ -> "1792x768" (~21:9).

    Returns:
        Path to the saved image file.
    """
    client = OpenAI()
    size = "1024x1360" if all_actions == 1 else "1536x1024" if all_actions == 2 else "1792x768"
    # 3:4, 3:2, 21:9
    response = client.images.generate(
        model=os.getenv("OPENAI_IMAGE_MODEL"),
        prompt=prompt,
        size=size,
        n=1,
    )
    image_bytes = base64.b64decode(response.data[0].b64_json)

    target_path = os.path.join(f"book_illustrations/{book_name}/", str(uuid.uuid1()) + filename)
    os.makedirs(f"book_illustrations/{book_name}", exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(image_bytes)
    return target_path