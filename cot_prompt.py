from llm_model import llm_model


def main():
    prompt = """Consider the problem:

A store had 22 apples. They sold 15 apples today and got a new delivery of 8 apples.
How many apples are there now?

Explain your reasoning step-by-step, then provide the final answer.
"""

    result = llm_model(prompt)
    print(result)


if __name__ == "__main__":
    main()
