from llm_model import llm_model


def main():
    prompt = """Here are a few examples of classifying emotions in statements:

Statement: "I just won my first marathon!"
Emotion: Joy

Statement: "I can't believe I lost my keys again."
Emotion: Frustration

Statement: "My best friend is moving to another country."
Emotion: Sadness

Now, classify the emotion in the following statement:
Statement: "That movie was so scary I had to cover my eyes."
Emotion:
"""

    result = llm_model(prompt)
    print(result)


if __name__ == "__main__":
    main()
