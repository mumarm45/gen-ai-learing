from llm_model import llm_model


def main():
    prompt = """Classify the following statement as true or false:

"The Eiffel Tower is located in Berlin."

Answer (true/false):
"""

    result = llm_model(prompt)
    print(result)


if __name__ == "__main__":
    main()
