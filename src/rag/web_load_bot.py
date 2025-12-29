from vectors.loader_vector import load_chroma

def web_load_bot(
    persist_dir: str = "./chroma_db",
    collection_name: str = "webresume",
    k: int = 4,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
    vector = load_chroma(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
    )
    retriever = vector.as_retriever(search_kwargs={"k": k})
    def search_documents(query, top_k=k):
        """Search for documents relevant to a query"""
        docs = retriever.invoke(query)
            
        # Limit to top_k if specified
        return docs[:top_k]
    test_queries = [
        "What is my name?",
        "What is my job?",
        "What is my location?"
        ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = search_documents(query)
            
        # Print the results
        print(f"Found {len(results)} relevant documents:")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}: {doc.page_content[:150]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")    
    return retriever


if __name__ == "__main__":
    web_load_bot()