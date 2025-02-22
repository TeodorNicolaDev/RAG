from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "RAG Pipeline API is running!"}