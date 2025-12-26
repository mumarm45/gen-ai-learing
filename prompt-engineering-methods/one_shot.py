from llm_model import llm_model


def main():
    prompt = """Here is an example of translating a sentence from English to Urdu:

English: "How is the weather today?"
Urdu: "aj ka weather kesa hai?"

Now, translate the following sentence from English to Urdu:

English: "Where is the nearest supermarket?"
Urdu:
"""

    result = llm_model(prompt)
    print(result)


if __name__ == "__main__":
    main()
