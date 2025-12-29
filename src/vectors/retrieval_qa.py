from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic

from .loader_vector import load_chroma
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def retrieval_qa(
    question: str,
    persist_dir: str = os.path.join(root, "chroma_db"),
    collection_name: str = "pdf",
    k: int = 4,
):
    vectorstore = load_chroma(persist_dir=persist_dir, collection_name=collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatAnthropic(model="claude-3-haiku-20240307")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    result = qa.invoke({"query": question})
    if isinstance(result, dict) and "result" in result:
        return result["result"]
    return result


if __name__ == "__main__":
    print(retrieval_qa("who is omar?"))