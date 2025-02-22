# app.py
from fastapi import FastAPI, File, UploadFile
from ingestion import extract_text_from_pdf, chunk_text
from query import is_trivial_query, transform_query, classify_query_with_mistral
from search import (
    get_embedding_from_mistral, 
    search_chunks, 
    merge_chunks,
    EMBEDDINGS_DB  # if needed for debugging or updating
)

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

@app.post("/query")
async def query_endpoint(query: dict):
    user_query = query.get("question", "")
    # First, check if the query is trivial
    if is_trivial_query(user_query):
        return {"response": "Hi there! How can I help you today?"}
    # Otherwise, optionally use Mistral to classify the query further
    query_type = classify_query_with_mistral(user_query)
    transformed = transform_query(user_query)
    # For now, just echo back details; later, integrate semantic search & generation
    return {
        "original_query": user_query,
        "query_type": query_type,
        "transformed_query": transformed,
        "response": "Proceeding with knowledge retrieval..."
    }