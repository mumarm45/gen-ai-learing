from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
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


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

def json_parser_chain():
    joke_query = "Tell me a joke."
    
    output_parser = JsonOutputParser(pydantic_object=Joke)

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm_model() | output_parser

    return chain.invoke({"query": joke_query})

if __name__ == "__main__":
    print(json_parser_chain())