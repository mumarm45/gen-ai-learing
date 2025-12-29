import os

if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "python-genai-langchain"
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
if "ANONYMIZED_TELEMETRY" not in os.environ:
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
if "CHROMA_TELEMETRY" not in os.environ:
    os.environ["CHROMA_TELEMETRY"] = "False"

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from langchain_chroma import Chroma
try:
    from .web_loader import web_loader
    from .pdf_loader import pdf_loader
    from .embeddings_model import embeddings_model
    from .splitter import recursive_text_splitter
except ImportError:  # pragma: no cover
    from web_loader import web_loader
    from pdf_loader import pdf_loader
    from embeddings_model import embeddings_model
    from splitter import recursive_text_splitter

def build_chroma_from_documents(
    documents,
    persist_dir: str = os.path.join(root, "chroma_db"),
    collection_name: str = "pdf",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
    if not documents:
        raise ValueError("No documents provided to build_chroma_from_documents")

    non_empty_documents = [d for d in documents if getattr(d, "page_content", None) and d.page_content.strip()]
    if not non_empty_documents:
        raise ValueError(
            "All provided documents are empty after loading/splitting. "
            "This would create empty embeddings and fail Chroma upsert."
        )

    os.makedirs(persist_dir, exist_ok=True)
    embeddings = embeddings_model(embedding_model_name)
    vectorstore = Chroma.from_documents(
        documents=non_empty_documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    persist = getattr(vectorstore, "persist", None)
    if callable(persist):
        persist()
    return vectorstore
def build_chroma_from_pdf(
    filepath: str,
    persist_dir: str = os.path.join(root, "chroma_db"),
    collection_name: str = "pdf",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
    chunks_recursive = pdf_loader(filepath)
    documents = chunks_recursive
    return build_chroma_from_documents(
        documents=documents,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
    )    
def build_chroma_from_web(
    url: str,
    persist_dir: str = os.path.join(root, "chroma_db"),
    collection_name: str = "web",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    verify_ssl: bool = True,
 ):
    docs = web_loader(url, verify_ssl=verify_ssl)
    if not docs:
        raise ValueError(f"No documents were loaded from URL: {url}")

    splitter = recursive_text_splitter()
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError(
            "Documents were loaded from the URL but splitting produced 0 chunks. "
            "Try a larger chunk_size, a different URL, or inspect the loaded HTML."
        )

    return build_chroma_from_documents(
        documents=chunks,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
    )


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Building vector store from PDF: {filepath}")
        build_chroma_from_pdf(filepath=filepath)
        print("Done!")
    else:
        print("No PDF file provided.")
        print("Usage: python build_vector.py <pdf_filepath>")

