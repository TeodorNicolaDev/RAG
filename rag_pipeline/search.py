import os
import numpy as np
from mistralai import Mistral

# Global dictionary to store embeddings for each text chunk.
# Format: {chunk_id: {"text": <text_chunk>, "embedding": <embedding_vector>}}
EMBEDDINGS_DB = {}

# Initialize Mistral client using the API key from environment variables.
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "r6cJTcdQ93jseOPcR8jlSgWDC87PmREn")
MODEL_NAME = "mistral-embed"
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# ------------------------------------------------------------------------------
# Function: get_embedding_from_mistral
# ------------------------------------------------------------------------------
def get_embedding_from_mistral(text: str):
    """
    Uses the Mistral Embeddings API to generate an embedding for the given text.
    
    :param text: The text to embed.
    :return: A numpy array representing the embedding vector (dimension 1024).
    """
    try:
        response = mistral_client.embeddings.create(
            model=MODEL_NAME,
            inputs=[text],
        )
        print("Mistral embedding API response:", response)
        # Extract the embedding from the response.
        # The response is an EmbeddingResponse object; we assume it has a 'data' attribute which is a list of Data objects.
        # Each Data object has an 'embedding' attribute that is a list of floats.
        embedding = response.data[0].embedding
        return np.array(embedding)
    except Exception as e:
        print(f"Error getting embedding from Mistral: {e}")
        return None

# ------------------------------------------------------------------------------
# Function: cosine_similarity
# ------------------------------------------------------------------------------
def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    
    :param vec1: A numpy array.
    :param vec2: A numpy array.
    :return: Cosine similarity score.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# ------------------------------------------------------------------------------
# Function: keyword_score
# ------------------------------------------------------------------------------
def keyword_score(chunk_text: str, query: str) -> float:
    """
    Computes a simple keyword matching score between the chunk and the query.
    
    :param chunk_text: The text chunk.
    :param query: The query text.
    :return: A normalized score based on keyword overlap.
    """
    query_words = set(query.lower().split())
    chunk_words = chunk_text.lower().split()
    if not chunk_words:
        return 0.0
    count = sum(1 for word in chunk_words if word in query_words)
    return count / len(chunk_words)

# ------------------------------------------------------------------------------
# Function: combined_score
# ------------------------------------------------------------------------------
def combined_score(semantic, keyword, alpha=0.7):
    """
    Combines the semantic and keyword scores using a weighted sum.
    
    :param semantic: Cosine similarity score.
    :param keyword: Keyword matching score.
    :param alpha: Weight for semantic similarity (default 0.7).
    :return: The combined score.
    """
    return alpha * semantic + (1 - alpha) * keyword

# ------------------------------------------------------------------------------
# Function: search_chunks
# ------------------------------------------------------------------------------
def search_chunks(query_embedding, query_text, top_n=5):
    """
    Searches the global EMBEDDINGS_DB for text chunks that best match the query.
    
    :param query_embedding: Numpy array for the query.
    :param query_text: The query text (for keyword matching).
    :param top_n: Number of top results to return.
    :return: List of tuples (chunk_id, chunk_text, combined_score).
    """
    results = []
    for chunk_id, data in EMBEDDINGS_DB.items():
        semantic = cosine_similarity(query_embedding, data["embedding"])
        keyword = keyword_score(data["text"], query_text)
        score = combined_score(semantic, keyword)
        results.append((chunk_id, data["text"], score))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

# ------------------------------------------------------------------------------
# Function: merge_chunks
# ------------------------------------------------------------------------------
def merge_chunks(chunks):
    """
    Merges a list of text chunks into a single context string.
    
    :param chunks: A list of text chunks.
    :return: A single string containing all chunks separated by newlines.
    """
    return "\n\n".join(chunks)