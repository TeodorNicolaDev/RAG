import requests
import numpy as np

# Global dictionary to store embeddings for each text chunk.
# Format: {chunk_id: {"text": <text_chunk>, "embedding": <embedding_vector>}}
EMBEDDINGS_DB = {}


# ------------------------------------------------------------------------------
# Function: get_embedding_from_mistral
# ------------------------------------------------------------------------------
MISTRAL_EMBEDDING_API_KEY = "gwGmq0uYH2PWtj3ZnuNpDXeCgtqaXnOf"
MISTRAL_EMBEDDING_API_URL = "https://api.mistral.ai/v1/embeddings"  # Hypothetical endpoint

def get_embedding_from_mistral(text: str):
    """
    Calls Mistral's embedding API to generate an embedding for the input text.
    
    :param text: The text for which to generate an embedding.
    :return: A list or numpy array representing the embedding vector.
    """
    headers = {
        "Authorization": f"Bearer {MISTRAL_EMBEDDING_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}
    try:
        response = requests.post(MISTRAL_EMBEDDING_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        embedding = response.json().get("embedding")
        # Convert to numpy array for similarity computation
        return np.array(embedding)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# ------------------------------------------------------------------------------
# Function: cosine_similarity
# ------------------------------------------------------------------------------
def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    
    :param vec1: First vector (numpy array).
    :param vec2: Second vector (numpy array).
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
    :param query: The user query.
    :return: A normalized keyword matching score.
    """
    query_words = set(query.lower().split())
    chunk_words = chunk_text.lower().split()
    if not chunk_words:
        return 0.0
    # Count of query words in the chunk
    count = sum(1 for word in chunk_words if word in query_words)
    return count / len(chunk_words)

# ------------------------------------------------------------------------------
# Function: combined_score
# ------------------------------------------------------------------------------
def combined_score(semantic, keyword, alpha=0.7):
    """
    Combines semantic and keyword scores using a weighted sum.
    
    :param semantic: The cosine similarity score.
    :param keyword: The keyword matching score.
    :param alpha: Weight for semantic similarity (default 0.7).
    :return: The combined score.
    """
    return alpha * semantic + (1 - alpha) * keyword


# ------------------------------------------------------------------------------
# Function: search_chunks
# ------------------------------------------------------------------------------
def search_chunks(query_embedding, query_text, top_n=5):
    """
    Searches the stored text chunks and returns the top_n results based on the combined score.
    
    :param query_embedding: Embedding vector for the query.
    :param query_text: The original query text (used for keyword matching).
    :param top_n: Number of top results to return.
    :return: A list of tuples (chunk_id, chunk_text, combined_score).
    """
    results = []
    for chunk_id, data in EMBEDDINGS_DB.items():
        # Compute semantic similarity using cosine similarity
        semantic = cosine_similarity(query_embedding, data["embedding"])
        # Compute keyword matching score
        keyword = keyword_score(data["text"], query_text)
        # Combine scores using our weighted function
        score = combined_score(semantic, keyword)
        results.append((chunk_id, data["text"], score))
    
    # Sort results by combined score in descending order
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]