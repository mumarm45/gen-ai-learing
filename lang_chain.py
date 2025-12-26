import os
import sys
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

def llm_model_langchain(prompt, params=None):

    load_dotenv()
    params = params or {}

    api_key = params.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY in your .env file")

    model = params.get("model", "claude-3-haiku-20240307")
    max_tokens = params.get("max_tokens", 400)
    temperature = params.get("temperature", 0.7)

   

    llm = ChatAnthropic(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    chain = llm | StrOutputParser()
    return chain.invoke(prompt)
