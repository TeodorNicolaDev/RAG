from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_basic_query():
    # Testing with a trivial query
    response = client.post("/query", json={"question": "hello"})
    assert response.status_code == 200
    data = response.json()
    # Expect a simple greeting response for trivial queries
    assert "Hi there" in data["response"]

def test_knowledge_query():
    # Testing with a more complex query
    response = client.post("/query", json={"question": "Tell me about quantum physics."})
    assert response.status_code == 200
    data = response.json()
    # Expect the endpoint to indicate knowledge retrieval
    assert data["query_type"] in {"basic", "knowledge"}
    assert "transformed_query" in data