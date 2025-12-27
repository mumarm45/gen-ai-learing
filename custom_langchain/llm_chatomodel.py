from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

def llm_model():
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY in your .env file")

    return ChatAnthropic(
        api_key=api_key,
        model="claude-3-haiku-20240307",
        temperature=0.7,
        max_tokens=400,
    )
