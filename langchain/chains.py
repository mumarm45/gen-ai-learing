import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage


def country_chain():
    template = """Your job is to come up with the classic dish from the given country.
    
    Country: {country}
    
    Classic dish: """
    
    prompt = PromptTemplate(template=template, input_variables=["country"])

    llm = LLMChain(llm=llm_model(), prompt=prompt, output_key="classic_dish")
    

    return llm 

def meal_chain():
    template = """Give me a meal {classic_dish} give a simple and short recipe on how to make it.
    
    Classic dish: {classic_dish}
    
    Recipe: """
    
    prompt = PromptTemplate(template=template, input_variables=["classic_dish"])

    llm = LLMChain(llm=llm_model(), prompt=prompt, output_key="recipe")
    
    return llm

def time_chain():
    template = """Give recipe {recipe} calculate the time it takes to make it.
    
    Recipe: {recipe}
    
    Time: """
    
    prompt = PromptTemplate(template=template, input_variables=["recipe"])

    llm = LLMChain(llm=llm_model(), prompt=prompt, output_key="time")
    
    return llm

def all_chains(): 

    country = country_chain()
    meal = meal_chain()
    time = time_chain()

    return SequentialChain(
        chains=[country, meal, time],
        input_variables=["country"],
        output_variables=["classic_dish", "recipe", "time"],
    )


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

def message_placeholder():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder("msgs")  # This will be replaced with one or more messages
    ])

    input_ = {"msgs": [HumanMessage(content="What is the day after Tuesday?")]}

    llm = llm_model()
    msg = llm.invoke(prompt.invoke(input_))
    print(msg.content)


if __name__ == "__main__":
    # print(all_chains().invoke({"country": "pakistan"}))
    message_placeholder()