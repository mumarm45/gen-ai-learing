from langchain_huggingface import HuggingFaceEmbeddings
import os


if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

if __name__ == "__main__":
    embeddings_model()    