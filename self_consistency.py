from llm_model import llm_model


def main():
    prompt = """When I was 6, my sister was half of my age. Now I am 70, what age is my sister?

Provide three independent calculations, then choose the most consistent final answer.
"""

    result = llm_model(prompt)
    print(result)


if __name__ == "__main__":
    main()
