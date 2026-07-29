import uuid

from dotenv import load_dotenv
load_dotenv()

from src.graph.main_graph import graph


config = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = graph.invoke(
    {"PROMPT": """
Topic: The Last Kid in Mythology Camp
    """, 
     "currentPage": 0, "num_pages": 1, "author":"Olivertwist", "book_name":"lastkid"},
    config=config,
)
for page in result["PAGES"]:
    print(page["page_number"], page["topic"], "->", page["image_path"])
