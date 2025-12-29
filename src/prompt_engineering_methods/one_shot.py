from llm_model import llm_model


def generate_response_one_shot(prompt_txt):
    template = """Here is an example of translating a sentence from English to Urdu:

English: "How is the weather today?"
Urdu: "aj ka weather kesa hai?"

Now, translate the following sentence from English to Urdu:

    English: "{question}"
    Urdu: 
"""
    final_prompt = template.format(question=prompt_txt)
    result = llm_model(final_prompt)
    return result


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
