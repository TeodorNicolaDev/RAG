from fastapi import FastAPI, File, UploadFile
from ingestion import process_pdf_and_store_embeddings
from query import is_trivial_query, transform_query, classify_query_with_mistral
from search import get_embedding_from_mistral, search_chunks, merge_chunks

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "RAG Pipeline API is running!"}

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    """
    Endpoint to ingest PDF files. It extracts text, chunks the text, generates embeddings,
    and stores them in a global in-memory dictionary.
    """
    total_chunks = 0
    details = {}
    for file in files:
        chunks_processed = process_pdf_and_store_embeddings(file)
        total_chunks += chunks_processed
        details[file.filename] = chunks_processed
    return {"message": "Files processed and embeddings stored", "total_chunks": total_chunks, "details": details}

@app.post("/query")
async def query_endpoint(query: dict):
    """
    Endpoint to process user queries. It determines if the query is trivial,
    transforms and classifies the query, generates its embedding using Mistral,
    searches for relevant text chunks, and returns a merged context.
    """
    user_query = query.get("question", "")
    if is_trivial_query(user_query):
        return {"response": "Hi there! How can I help you today?"}
    
    # Use our (stub) classification and transformation functions.
    query_type = classify_query_with_mistral(user_query)
    transformed = transform_query(user_query)
    
    # Generate the query embedding using the Mistral API.
    query_embedding = get_embedding_from_mistral(transformed)
    if query_embedding is None:
        return {"response": "Error generating query embedding."}
    
    # Retrieve the top matching text chunks from the global embedding store.
    results = search_chunks(query_embedding, transformed)
    top_chunks = [res[1] for res in results]
    context = merge_chunks(top_chunks)
    
    return {
        "original_query": user_query,
        "query_type": query_type,
        "transformed_query": transformed,
        "retrieved_context": context,
        "response": "Proceeding with knowledge retrieval..."
    }