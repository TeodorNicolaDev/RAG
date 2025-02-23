import os
import numpy as np
import faiss
from mistralai import Mistral

# Global lists for storing text chunks and embeddings.
TEXT_CHUNKS = []
EMBEDDINGS_LIST = []

# Set up the Mistral embedding client.
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "r6cJTcdQ93jseOPcR8jlSgWDC87PmREn")
MODEL_NAME = "mistral-embed"
client = Mistral(api_key=MISTRAL_API_KEY)

def get_embedding_from_mistral(text: str):
    """
    Uses Mistral's Embeddings API to generate a 1024-dimensional embedding.
    """
    try:
        response = client.embeddings.create(
            model=MODEL_NAME,
            inputs=[text]
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype="float32")
    except Exception as e:
        print(f"Error getting embedding from Mistral: {e}")
        return None

def add_chunk(text: str, embedding: np.array):
    """
    Adds a text chunk and its embedding to the global lists.
    """
    TEXT_CHUNKS.append(text)
    EMBEDDINGS_LIST.append(embedding)

def build_faiss_index():
    """
    Builds and returns a Faiss index from stored embeddings.
    """
    if not EMBEDDINGS_LIST:
        return None
    embeddings_np = np.vstack(EMBEDDINGS_LIST).astype("float32")
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    return index

def simple_keyword_score(chunk_text: str, query_text: str) -> float:
    """
    Computes a simple keyword matching score based on word overlap.
    """
    query_words = set(query_text.lower().split())
    chunk_words = set(chunk_text.lower().split())
    if not query_words:
        return 0.0
    return len(query_words.intersection(chunk_words)) / len(query_words)

def search_chunks(query_embedding, query_text, candidate_k=10, final_k=5, alpha=0.7):
    """
    Performs two-stage retrieval:
      1. Retrieve candidate_k chunks via Faiss using L2 distance.
      2. Re-rank candidates based on a weighted combination of cosine similarity and keyword matching.
    Returns the top final_k chunks.
    """
    index = build_faiss_index()
    if index is None:
        return []
    
    query_embedding = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_embedding, candidate_k)
    
    candidate_scores = []
    for idx in indices[0]:
        if idx < len(TEXT_CHUNKS):
            # Compute cosine similarity manually.
            chunk_embedding = EMBEDDINGS_LIST[idx]
            sem_sim = np.dot(query_embedding, np.array([chunk_embedding]).T)[0][0] / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            key_sim = simple_keyword_score(TEXT_CHUNKS[idx], query_text)
            combined = alpha * sem_sim + (1 - alpha) * key_sim
            candidate_scores.append((TEXT_CHUNKS[idx], combined))
    
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    final_chunks = [chunk for chunk, score in candidate_scores[:final_k]]
    return final_chunks

def merge_chunks(chunks):
    """
    Merges a list of text chunks into a single coherent context string.
    """
    return "\n\n".join(chunks)