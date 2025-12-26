import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

def llm_model_langchain(template, template_variables, params=None):

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


    prompt = PromptTemplate.from_template(template)
    def format_prompt(variables):
      return prompt.format(**variables)
    joke_chain = (
    RunnableLambda(format_prompt)
    | llm 
    | StrOutputParser()
   )

    result = joke_chain.invoke(template_variables)
    return result

def second_example(): 
    result = llm_model_langchain("Tell me a {adjective} joke about {content}.", 
    {"adjective": "funny", "content": "cats"})
    return result

def question_answering_example():
    content = """
    The solar system consists of the Sun, eight planets, their moons, dwarf planets, and smaller objects like asteroids and comets. 
    The inner planets—Mercury, Venus, Earth, and Mars—are rocky and solid. 
    The outer planets—Jupiter, Saturn, Uranus, and Neptune—are much larger and gaseous.
    """
    question = "Which planets in the solar system are rocky and solid?"
    template = """
    Answer the {question} based on the {content}.
    Respond "Unsure about answer" if not sure about the answer.
    
    Answer:
    
  """
    result = llm_model_langchain(template, 
    {"question": question, "content": content})
    return result

def text_classification():
    text = """
    The concert last night was an exhilarating experience with outstanding performances by all artists.
    """
    categories = "Entertainment, Food and Dining, Technology, Literature, Music."
    template = """
    Classify the {text} into one of the {categories}.
    
    Category:
    
    """
    result = llm_model_langchain(template, 
    {"text": text, "categories": categories})
    return result
    
def code_generation():
    description = """
    Retrieve the names and email addresses of all customers from the 'customers' table who have made a purchase in the last 30 days. 
    The table 'purchases' contains a column 'purchase_date'
    """
    template = """
    {description}
    
    Generate the code:
    
    """
    result = llm_model_langchain(template, 
    {"description": description})
    return result

def role_play():
    role = """
    Software Engineer
    """
    tone = "engaging and immersive"
    template = """
    You are an expert {role}. I have this question {question}. I would like our conversation to be {tone}.
    give ans in 1 sentence
    Answer:
    
    """    
    while True:
        query = input("Question: ")
    
        if query.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break

        result = llm_model_langchain(template, 
        {"role": role, "tone": tone, "question": query})
        print("Answer:", result)
    return result

if __name__ == "__main__":
    role_play()