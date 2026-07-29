from src.graph.agent_state import BookPage, BookState
from langgraph.graph import START, END, StateGraph
from src.graph.agents import (
    researcher_node, outline_node_conversation, cover_node,
    page_node_conversation, emphasis_node, pdf_writer_node,
    route_after_page
)
from langgraph.checkpoint.memory import MemorySaver

# Graph: researcher -> outliner -> cover -> page_node (loop over PAGE_OUTLINE) -> emphasis -> pdf_writer
builder = StateGraph(BookState)
builder.add_node("researcher", researcher_node)
builder.add_node("outliner", outline_node_conversation)
builder.add_node("cover", cover_node)
builder.add_node("page_node_conversation", page_node_conversation)
builder.add_node("emphasis", emphasis_node)
builder.add_node("pdf_writer", pdf_writer_node)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", "outliner")
builder.add_edge("outliner", "cover")
builder.add_edge("cover", "page_node_conversation")
builder.add_conditional_edges("page_node_conversation", route_after_page, ["page_node_conversation", "emphasis"])
builder.add_edge("emphasis", "pdf_writer")
builder.add_edge("pdf_writer", END)

graph = builder.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    from pathlib import Path

    graph_png_path = Path(__file__).resolve().parents[3] / "docs" / "graph.png"
    graph_png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        graph_png_path.write_bytes(graph.get_graph().draw_mermaid_png())
        print(f"Saved graph diagram to {graph_png_path}")
    except Exception as e:
        print(f"Could not save graph: {e}")