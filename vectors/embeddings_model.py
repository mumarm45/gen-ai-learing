from langchain_huggingface import HuggingFaceEmbeddings
import os
import sys
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)
    
def parent_document_retriever(filepath):
    if not filepath:
        raise ValueError("filepath is required")
    if not os.path.exists(filepath):
        raise ValueError(f"File path {filepath} is not a valid file or url")

    store = InMemoryStore()
    loader = PyPDFLoader(filepath)
    documents = loader.load()

    if not documents:
        raise ValueError("No pages were loaded from the PDF")

    non_empty_documents = [d for d in documents if (d.page_content or "").strip()]
    if not non_empty_documents:
        raise ValueError(
            "PDF loaded, but no text was extracted. The PDF may be scanned images and require OCR."
        )

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    vectorstore = Chroma (
    collection_name="split_parents", embedding_function=embeddings_model()
    )
    retriever = ParentDocumentRetriever(
    # The vector store where child document embeddings will be stored and searched
    # This Chroma instance will contain the embeddings for the smaller chunks
    vectorstore=vectorstore,
    
    # The document store where parent documents will be stored
    # These larger chunks won't be embedded but will be retrieved by ID when needed
    docstore=store,
    
    # The splitter used to create small chunks (400 chars) for precise vector search
    # These smaller chunks are embedded and used for similarity matching
    child_splitter=child_splitter,
    
    # The splitter used to create larger chunks (2000 chars) for better context
    # These parent chunks provide more complete information when retrieved
    parent_splitter=parent_splitter,
    )
    retriever.add_documents(non_empty_documents)

    print(len(list(store.yield_keys())))
    sub_docs = vectorstore.similarity_search("omar")
    print(sub_docs[0].page_content)
    retrieved_docs = retriever.invoke("omar")
    print(retrieved_docs[0].page_content)
    return retriever
    

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "resume.pdf"
    parent_document_retriever(filepath)