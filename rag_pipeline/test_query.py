import time
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_query_endpoint_basic():
    # Pause before making the API call to avoid rate limits.
    time.sleep(2)
    
    response = client.post("/query", json={"question": "hello"})
    print("Response from /query with trivial query:")
    print(response.text)
    
    assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"
    data = response.json()
    # For trivial queries, expect a simple greeting.
    assert "Hi there" in data["response"]

def test_query_endpoint_knowledge():
    # Ensure there is data in the embeddings DB.
    # If EMBEDDINGS_DB is empty, simulate ingestion.
    from ingestion import process_pdf_and_store_embeddings
    import tempfile
    import fitz
    if not hasattr(__import__("search"), "EMBEDDINGS_DB") or not __import__("search").EMBEDDINGS_DB:
        print("EMBEDDINGS_DB is empty. Populating via simulated ingestion...")
        def create_temp_pdf():
            temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Quantum physics is the study of matter and energy at the smallest scales.")
            doc.save(temp_pdf.name)
            doc.close()
            return temp_pdf.name
        sample_pdf_path = create_temp_pdf()
        print(f"Created sample PDF for knowledge query at: {sample_pdf_path}")
        with open(sample_pdf_path, "rb") as pdf_file:
            process_pdf_and_store_embeddings(pdf_file)
    
    # Pause to allow for any rate limiting issues.
    time.sleep(2)
    
    response = client.post("/query", json={"question": "Tell me about quantum physics."})
    print("Response from /query with knowledge query:")
    print(response.text)
    
    assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"
    data = response.json()
    
    # Check that the expected keys are present.
    assert "original_query" in data, "Missing 'original_query' key in response."
    assert "transformed_query" in data, "Missing 'transformed_query' key in response."
    assert "retrieved_context" in data, "Missing 'retrieved_context' key in response."
    assert len(data["retrieved_context"]) > 0, "Expected retrieved context to be non-empty."