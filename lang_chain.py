import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

def llm_model_langchain(params=None):

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

    template = """Tell me a {adjective} joke about {content}.
"""
    prompt = PromptTemplate.from_template(template)
    def format_prompt(variables):
      return prompt.format(**variables)
    joke_chain = (
    RunnableLambda(format_prompt)
    | llm 
    | StrOutputParser()
)

    result = joke_chain.invoke({"adjective": "sad", "content": "dogs"})
    return result


if __name__ == "__main__":
    result = llm_model_langchain()
    print(result)