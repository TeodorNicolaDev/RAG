import re

def is_trivial_query(query: str) -> bool:
    """
    Checks if the query is a trivial greeting.
    
    :param query: The input query.
    :return: True if the query is trivial, False otherwise.
    """
    trivial_keywords = {"hello", "hi", "hey", "good morning", "good evening"}
    normalized = query.lower().strip()
    return any(normalized.startswith(word) for word in trivial_keywords)

def transform_query(query: str) -> str:
    """
    Normalizes the query by removing punctuation and converting to lower case.
    
    :param query: The input query.
    :return: The normalized query.
    """
    return re.sub(r'[^\w\s]', '', query).lower().strip()

def classify_query_with_mistral(query: str) -> str:
    """
    Stub for classifying a query. In a real implementation, you might use an LLM or other logic.
    
    :param query: The input query.
    :return: 'knowledge' for non-trivial queries.
    """
    # For now, any non-trivial query is classified as 'knowledge'
    return "knowledge"