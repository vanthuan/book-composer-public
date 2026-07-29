from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from src.graph.llm_models import model
def _extract_text(content) -> str:
    """Normalize an AIMessage.content into plain text.

    Some models/integrations (e.g. this Gemini setup) return content as a
    list of content blocks (`[{"type": "text", "text": "..."}]`) instead of
    a plain string; downstream code (state fields, create_pdf_book) expects str.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def run_agent(system_prompt: str, tools: list, max_tool_iterations: int = 5) -> str:
    """Minimal manual ReAct loop: call the model, execute any tool calls,
    feed results back, and repeat until the model answers without calling a tool."""
    bound_model = model.bind_tools(tools) if tools else model
    tools_by_name = {t.name: t for t in tools}
    messages: list = [SystemMessage(content=system_prompt), HumanMessage(content="Proceed.")]

    ai_message = None
    for _ in range(max_tool_iterations):
        ai_message = bound_model.invoke(messages)
        if not ai_message.tool_calls:
            return _extract_text(ai_message.content)
        messages.append(ai_message)
        for tool_call in ai_message.tool_calls:
            result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return _extract_text(ai_message.content) if ai_message is not None else ""


