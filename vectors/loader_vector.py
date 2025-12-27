import os

if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "python-genai-langchain"
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from .embeddings_model import embeddings_model
def load_chroma(
    persist_dir: str = "./chroma_db",
    collection_name: str = "docs",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    embeddings = embeddings_model(embedding_model_name)
    try:
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    except KeyError as e:
        if str(e).strip("\"'") == "_type":
            raise RuntimeError(
                "Your existing Chroma persisted DB appears to be from an incompatible ChromaDB version. "
                "Delete the persist directory (default: ./chroma_db) and rebuild the index."
            ) from e
        raise

