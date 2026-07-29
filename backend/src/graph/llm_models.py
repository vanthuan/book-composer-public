import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# using google gemini
model_name = os.environ["GEMINI_MODEL"]

model = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.9,  # Gemini 3.0+ defaults to 1.0
    max_tokens=10000,
    timeout=None,
    max_retries=2,
    
)
# using open AI
# model = ChatOpenAI(model="gpt-5.1", temperature=0.8,  max_retries=2)
