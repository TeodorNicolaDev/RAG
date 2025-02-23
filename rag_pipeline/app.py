from fastapi import FastAPI, File, UploadFile
from rag_pipeline.ingestion import process_pdf_and_store_embeddings
from rag_pipeline.query import is_trivial_query, transform_query, classify_query_with_mistral
from rag_pipeline.search import get_embedding_from_mistral, search_chunks, merge_chunks
from rag_pipeline.generation import generate_answer

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "RAG Pipeline API is running!"}

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    total_chunks = 0
    details = {}
    for file in files:
        chunks_processed = process_pdf_and_store_embeddings(file)
        total_chunks += chunks_processed
        details[file.filename] = chunks_processed
    return {"message": "Files processed and embeddings stored", "total_chunks": total_chunks, "details": details}

@app.post("/query")
async def query_endpoint(query: dict):
    user_query = query.get("question", "")
    
    if is_trivial_query(user_query):
        # For trivial queries, generate a direct response.
        answer = generate_answer("", user_query)
        return {"response": answer}
    
    query_type = classify_query_with_mistral(user_query)
    transformed = transform_query(user_query)
    
    query_embedding = get_embedding_from_mistral(transformed)
    if query_embedding is None:
        return {"response": "Error generating query embedding."}
    
    retrieved_chunks = search_chunks(query_embedding, transformed)
    context = merge_chunks(retrieved_chunks)
    
    answer = generate_answer(context, user_query)
    
    return {
        "original_query": user_query,
        "query_type": query_type,
        "transformed_query": transformed,
        "retrieved_context": context,
        "generated_answer": answer
    }