


from vectors.loader_vector import load_chroma
import os
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain



root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_prompt_template():
    """
    Create a prompt template for the QA chain.
    
    Returns:
        PromptTemplate: The prompt template to use for document-based QA.
    """
    template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
If asked to summarize, provide a concise summary.

{context}

Question: {question}
Helpful Answer:"""
    return PromptTemplate(template=template, input_variables=["context", "question"])
    
def get_anthropic(model: str = "claude-3-haiku-20240307"):
    """
    Create and return a ChatAnthropic instance.
    
    Args:
        model (str): The Anthropic model to use. Defaults to "claude-3-haiku-20240307".
        
    Returns:
        ChatAnthropic: An instance of the ChatAnthropic class.
        
    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set in environment variables.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY in your .env file")
    
    return ChatAnthropic(model=model, api_key=api_key)

def get_vector_retriever(persist_dir: str = os.path.join(root, "chroma_db"), collection_name: str = "pdf", k: int = 10):
    """
    Load Chroma vector store and return retriever.
    
    Args:
        persist_dir (str): Directory where Chroma database is stored.
        collection_name (str): Name of the Chroma collection.
        k (int): Number of documents to retrieve.
        
    Returns:
        tuple: (vectorstore, retriever) tuple.
    """
    vectorstore = load_chroma(persist_dir=persist_dir, collection_name=collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return vectorstore, retriever

def retrieval_qa(
    question: str,
    history: list = None,
    persist_dir: str = os.path.join(root, "chroma_db"),
    collection_name: str = "pdf",
    k: int = 10,
    debug: bool = False,
    ):
    """
    Perform retrieval-based question answering.
    
    Args:
        question (str): The question to answer.
        history (list, optional): Conversation history.
        persist_dir (str): Directory where Chroma database is stored.
        collection_name (str): Name of the Chroma collection.
        k (int): Number of documents to retrieve.
        debug (bool): Whether to print debug information.
        
    Returns:
        str: The answer to the question.
    """
    load_dotenv()

    llm = get_anthropic()
    _, retriever = get_vector_retriever(persist_dir=persist_dir, collection_name=collection_name, k=k)
    
    # Create memory with proper configuration
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    
    prompt_template = create_prompt_template()

    # Create the conversational retrieval chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        verbose=False
    )

    # Invoke the chain
    result = qa.invoke({"question": question})
    
    # Extract answer from result
    if isinstance(result, dict) and "answer" in result:
        return result["answer"]
    return str(result)


if __name__ == "__main__":
    import sys
    
    # Check if --debug flag is passed
    debug_mode = "--debug" in sys.argv
    
    history = []
    print("RAG Question Answering System")
    print("Type 'quit', 'exit', or 'bye' to exit")
    while True:
        query = input("\nQuestion: ")
        
        if query.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break
            
        try:
            result = retrieval_qa(question=query, history=history, debug=debug_mode)
            
            history.append((query, result))
            
            print("\nAnswer:", result)
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'bye' to exit")
