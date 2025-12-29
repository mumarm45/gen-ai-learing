from fastapi import FastAPI, Request
from fastapi import HTTPException
from rag.rag_answer import rag_answer

app = FastAPI()

"""
GenAI RAG API
Provides endpoints for asking questions and checking health.

curl example:
curl -X POST http://localhost:4200/ask -H "Content-Type: application/json" -d '{"question": "What is the capital of France?"}'
"""
@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get('question', '')
    
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    
    try:
        answer = rag_answer(question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"message": "GenAI RAG API is running", "version": "1.0.0"}

def main():
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=4200, log_level="info")

if __name__ == '__main__':
    main()