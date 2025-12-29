from vectors.loader_vector import load_chroma
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_vector(
    persist_dir: str = os.path.join(root, "chroma_db"),
    collection_name: str = "pdf",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    vectorstore = load_chroma(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
    )   
    docs = vectorstore.similarity_search("is he work on react", k=1)
    return docs

if __name__ == "__main__":
    print(test_vector())