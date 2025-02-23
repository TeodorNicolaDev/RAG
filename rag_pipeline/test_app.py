from fastapi.testclient import TestClient
from rag_pipeline.app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "RAG Pipeline API is running!"}