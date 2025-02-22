import re
import requests

# ------------------------------------------------------------------------------
# Function: is_trivial_query
# ------------------------------------------------------------------------------
def is_trivial_query(query: str) -> bool:
    """
    Checks if the query is a simple greeting or casual remark.
    Returns True if it is trivial, False otherwise.
    """
    trivial_keywords = {"hello", "hi", "hey", "good morning", "good evening"}
    normalized_query = query.lower().strip()
    # If query is exactly a greeting or starts with one, consider it trivial
    return any(normalized_query.startswith(keyword) for keyword in trivial_keywords)

# ------------------------------------------------------------------------------
# Function: transform_query
# ------------------------------------------------------------------------------
def transform_query(query: str) -> str:
    """
    Transforms and normalizes the query by removing punctuation and extra spaces.
    """
    # Remove punctuation and extra whitespace, then lower-case the query
    return re.sub(r'[^\w\s]', '', query).lower().strip()

# ------------------------------------------------------------------------------
# Function: classify_query_with_mistral
# ------------------------------------------------------------------------------
MISTRAL_API_KEY = "gwGmq0uYH2PWtj3ZnuNpDXeCgtqaXnOf"
MISTRAL_API_URL = "https://api.mistral.ai/v1/generate"

def classify_query_with_mistral(query: str) -> str:
    """
    Uses the Mistral AI API to classify a query.
    Returns 'basic' if the query is a trivial greeting, or 'knowledge' if it requires knowledge retrieval.
    """
    prompt = (
        f"Classify the following query as either 'basic' if it is a greeting or simple remark, "
        f"or 'knowledge' if it requires detailed information: \n\n"
        f"Query: '{query}'\n\nResponse:"
    )
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"prompt": prompt, "max_tokens": 5}
    try:
        response = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        classification = response.json().get("answer", "").strip().lower()
        # Return classification based on the API response
        if "basic" in classification:
            return "basic"
        return "knowledge"
    except Exception as e:
        print(f"Error classifying query: {e}")
        # Fallback to knowledge if there's an error
        return "knowledge"