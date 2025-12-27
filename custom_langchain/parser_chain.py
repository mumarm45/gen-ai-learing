from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from custom_langchain.llm_chatomodel import llm_model


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

def comma_separated_list_parser_chain():
    joke_query = "Tell me a joke."
    
    output_parser = CommaSeparatedListOutputParser()

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm_model() | output_parser

    return chain.invoke({"query": joke_query})

def json_output_chain2():
    output_parser = JsonOutputParser()
    format_instructions = """RESPONSE FORMAT: Return ONLY a single JSON object—no markdown, no examples, no extra keys.  It must look exactly like:
    {
     "title": "movie title",
     "director": "director name",
      "year": 2000,
      "genre": "movie genre"
    }"""

    prompt_template = PromptTemplate(
    template="""You are a JSON-only assistant.

    Task: Generate info about the movie "{movie_name}" in JSON format.

    {format_instructions}
    """,
    input_variables=["movie_name", "format_instructions"],
    )

    chain = prompt_template | llm_model() | output_parser
    return chain.invoke({"movie_name": "Inception", "format_instructions": format_instructions})
if __name__ == "__main__":
    print(json_output_chain2())