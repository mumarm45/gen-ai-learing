import os

if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "python-genai-langchain"
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from web_loader import web_loader
from pdf_loader import pdf_loader
from embeddings_model import embeddings_model
from splitter import recursive_text_splitter

def build_chroma_from_documents(
    documents,
    persist_dir: str = "../chroma_db",
    collection_name: str = "docs",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = embeddings_model(embedding_model_name)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    return vectorstore
def build_chroma_from_pdf(
    filepath: str,
    persist_dir: str = "../chroma_db",
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
    persist_dir: str = "../chroma_db",
    collection_name: str = "web",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
 ):
    docs = web_loader(url)
    splitter = recursive_text_splitter()
    chunks = splitter.split_documents(docs)
    return build_chroma_from_documents(
        documents=chunks,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
    )
