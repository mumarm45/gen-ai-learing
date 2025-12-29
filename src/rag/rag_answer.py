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
    docs = use_similarity_search(question, k)
    print(docs[0].metadata)
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
                """
                Context:{context}
                
                Question: {question}

                Answer:
                """,
            ),
        ]
    )

    chain = prompt | RunnableLambda(lambda v: llm_model(v.to_string())) | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

def use_retrievers(question, k):
    vectorstore = load_chroma(persist_dir="../chroma_db", collection_name="pdf")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)
def  use_similarity_search(question, k):
    vectorstore = load_chroma(persist_dir="../chroma_db", collection_name="pdf")
    docs = vectorstore.similarity_search(question, k=k)
    return docs  

if __name__ == "__main__":
    print(rag_answer())