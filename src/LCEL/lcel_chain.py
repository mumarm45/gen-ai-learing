import os
import sys
from pprint import pprint



from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from llm_model import llm_model

# Define the templates for each step
location_template = """Your job is to come up with a classic dish from the area that the users suggests.
{location}

YOUR RESPONSE:
"""

dish_template = """Given a meal {meal}, give a short and simple recipe on how to make that dish at home.

YOUR RESPONSE:
"""

time_template = """Given the recipe {recipe}, estimate how much time I need to cook it.

YOUR RESPONSE:
"""


llm_runnable = RunnableLambda(lambda v: llm_model(v.to_string()))

location_chain_lcel = (
    PromptTemplate.from_template(location_template)  
    | llm_runnable
    | StrOutputParser()                              
)

dish_chain_lcel = (
    PromptTemplate.from_template(dish_template)      
    | llm_runnable
    | StrOutputParser()                              
)

time_chain_lcel = (
    PromptTemplate.from_template(time_template)      
    | llm_runnable
    | StrOutputParser()                              
)

overall_chain_lcel = (
    RunnablePassthrough.assign(meal=location_chain_lcel)
    | RunnablePassthrough.assign(recipe=dish_chain_lcel)
    | RunnablePassthrough.assign(time=time_chain_lcel)
)
result = overall_chain_lcel.invoke({"location": "China"})
pprint(result)