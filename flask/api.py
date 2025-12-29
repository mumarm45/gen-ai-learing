from fastapi import FastAPI, Request
from rag.rag_answer import rag_answer

app = FastAPI()

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get('question', '')
    
    if not question:
        return {"error": "No question provided"}, 400
    
    try:
        answer = rag_answer(question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000, log_level="info")