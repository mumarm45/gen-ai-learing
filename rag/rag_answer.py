import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vectors.loader_vector import load_chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from llm_model import llm_model

def rag_answer(
    question: str = "Who is Omar?",
    persist_dir: str = "../chroma_db",
    collection_name: str = "pdf",
    k: int = 4,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    vectorstore = load_chroma(persist_dir=persist_dir, 
    collection_name=collection_name, embedding_model_name=embedding_model_name)
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[Chunk {i + 1}]\n{d.page_content}" for i, d in enumerate(docs)
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful HR assistant. Answer using only the provided context. If the context is insufficient, say you don't know.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            ),
        ]
    )

    chain = prompt | RunnableLambda(lambda v: llm_model(v.to_string())) | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

if __name__ == "__main__":
    print(rag_answer())