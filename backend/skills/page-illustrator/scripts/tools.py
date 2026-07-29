from langchain_core.tools import tool

from src.illustrator.image_model import call_image_model


def make_tools(updates: dict, *, book_name: str, **_vars) -> list:
    """Build this call's sink tools, bound to `updates` and `book_name`.

    Same tool-as-sink convention as the notebook's make_generate_image_tool:
    the tool writes its result into `updates` as a side effect and returns
    only a status ack to the model.
    """
    @tool
    def generate_image(prompt: str, filename: str, all_actions: int) -> dict:
        """Generate the page's illustration and save it locally.

        Args:
            prompt: Image generation prompt. Flat illustration style, no text in the image.
            filename: Unique filename to save as, e.g. page_01.png
            all_actions: Number of left-to-right moments/actions composed
                into this one image — drives the aspect ratio requested
                from the image model (wider for more moments).
        """
        path = call_image_model(prompt, book_name, filename, all_actions)
        updates.setdefault("image_path", []).append(path)
        return {"status": "success", "image_path": path}

    return [generate_image]