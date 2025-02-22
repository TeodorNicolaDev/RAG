# app.py
from fastapi import FastAPI, File, UploadFile
from ingestion import extract_text_from_pdf, chunk_text

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "RAG Pipeline API is running!"}

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    total_chunks = 0
    details = {}
    for file in files:
        text = extract_text_from_pdf(file)
        chunks = chunk_text(text)
        total_chunks += len(chunks)
        details[file.filename] = len(chunks)
    return {"message": "Files processed", "total_chunks": total_chunks, "details": details}