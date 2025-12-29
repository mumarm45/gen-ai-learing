
from vectors.build_vector import build_chroma_from_documents
from vectors.pdf_loader import pdf_loader
from vectors.loader_vector import load_chroma
from vectors.retrieval_qa import get_anthropic, create_prompt_template



import gradio as gr
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
load_dotenv()
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load_document_chroma(file_input):
    # Load and process the document using Chroma
    # This is a placeholder - you'll need to implement the actual logic
    
    
    
    file_path = file_input

    # Load the PDF
    chunks = pdf_loader(file_path)
    
    
    print(f"Loaded {len(chunks)} chunks from PDF")
    if chunks:
        print(f"First chunk preview: {chunks[0].page_content[:200]}...")
    
    # Build Chroma vector store
    persist_path = os.path.join(root, "chroma_db")
    vectorstore = build_chroma_from_documents(documents=chunks, persist_dir=persist_path, collection_name="contract_docs")
    
    print(f"Created Chroma vector store with {len(vectorstore.get()['ids'])} documents")
    return vectorstore

def get_vectorstore():
    return load_chroma(collection_name="contract_docs")


def retriever_qa(file, query):
    template = create_prompt_template()
    vectorstore = load_document_chroma(file)
    llm = get_anthropic()  
    retriever_obj = vectorstore.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
                                    llm=llm, 
                                    retriever=retriever_obj, 
                                    combine_docs_chain_kwargs={"prompt": template},
                                    return_source_documents=True,
                                    memory=None,
                                    verbose=False
                                    )
    response = qa.invoke({"question": query, "chat_history": []})
    return response['answer']

rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PDF Q&A Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

if __name__ == "__main__":
    rag_application.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False, 
        debug=True, 
        inbrowser=True, 
        quiet=False
    )