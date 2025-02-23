import re

def is_trivial_query(query: str) -> bool:
    """
    Determines if a query is trivial (e.g., greetings).
    """
    trivial_keywords = {"hello", "hi", "hey", "good morning", "good evening"}
    normalized = query.lower().strip()
    return any(normalized.startswith(word) for word in trivial_keywords)

def transform_query(query: str) -> str:
    """
    Normalizes the query by removing punctuation and extra spaces.
    """
    return re.sub(r'[^\w\s]', '', query).lower().strip()

def classify_query_with_mistral(query: str) -> str:
    """
    A stub classification function that always returns 'knowledge' for non-trivial queries.
    """
    return "knowledge"