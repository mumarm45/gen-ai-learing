from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from tools.calculator import tools as build_tools
from custom_langchain.llm_chatomodel import llm_model

def bot_for_tol():
    prompt_template = """You are an agent who has access to the following tools:

    {tools}

    The available tools are: {tool_names}

    To use a tool, please use the following format:
    ```
    Thought: I need to figure out what to do
    Action: tool_name
    Action Input: the input to the tool
    ```

    After you use a tool, the observation will be provided to you:
    ```
    Observation: result of the tool
    ```

    Then you should continue with the thought-action-observation cycle until you have enough information to respond to the user's request directly.
    When you have the final answer, respond in this format:
    ```
    Thought: I know the answer
    Final Answer: the final answer to the original query
    ```

    Remember, when using the Python Calculator tool, the input must be valid Python code.

    Begin!

    Question: {input}
    {agent_scratchpad}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    tools_list = build_tools()
    agent = create_react_agent(
        llm=llm_model(),
        tools=tools_list,
        prompt=prompt,
    )
    return agent, tools_list

def run_agent():
    agent, tools_list = bot_for_tol()
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools_list, 
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor.invoke({"input": "What is 2+2?"}) 

if __name__ == "__main__":
    run_agent()