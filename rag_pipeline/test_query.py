import time
import tempfile
import fitz  # PyMuPDF
from fastapi.testclient import TestClient
from rag_pipeline.app import app
from rag_pipeline.ingestion import process_pdf_and_store_embeddings
from rag_pipeline.search import TEXT_CHUNKS, EMBEDDINGS_LIST

client = TestClient(app)

def create_long_pdf_with_authors():
    """
    Creates a temporary PDF with longer content that includes author information.
    """
    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "Authors: Alice, Bob.\n\n"
        "Quantum physics is a complex field that examines matter and energy at the smallest scales. "
        "This paper presents advanced research findings with detailed experiments and analysis. "
        "The research was conducted by a team of experts.\n\n"
        "Additional sections discuss methodology, data analysis, and future research directions. "
        "The paper provides insights into both theoretical models and experimental results."
    )
    page.insert_text((72, 72), text)
    doc.save(temp_pdf.name)
    doc.close()
    return temp_pdf.name

def test_query_endpoint_trivial():
    time.sleep(2)
    response = client.post("/query", json={"question": "hello"})
    print("Trivial Query Response:", response.text)
    assert response.status_code == 200
    data = response.json()
    trivial_response = data.get("response", "").lower()
    assert "hi there" in trivial_response or "hello" in trivial_response, \
           f"Expected a greeting, got: {data.get('response', '')}"

def test_query_endpoint_knowledge():
    # Clear global storage.
    TEXT_CHUNKS.clear()
    EMBEDDINGS_LIST.clear()
    
    print("Populating data via simulated ingestion with longer PDF content...")
    sample_pdf_path = create_long_pdf_with_authors()
    print(f"Created sample PDF at: {sample_pdf_path}")
    with open(sample_pdf_path, "rb") as pdf_file:
        process_pdf_and_store_embeddings(pdf_file)
    
    time.sleep(5)
    response = client.post("/query", json={"question": "Who are the authors of the paper?"})
    print("Knowledge Query Response:", response.text)
    assert response.status_code == 200
    data = response.json()
    assert "original_query" in data
    assert "transformed_query" in data
    assert "retrieved_context" in data
    assert "generated_answer" in data
    assert len(data["retrieved_context"]) > 0, "Expected non-empty retrieved context."
    assert len(data["generated_answer"]) > 0, "Expected non-empty generated answer."
    context_lower = data["retrieved_context"].lower()
    assert "alice" in context_lower or "bob" in context_lower, \
           "Expected retrieved context to mention the authors (Alice, Bob)."

if __name__ == "__main__":
    test_query_endpoint_trivial()
    test_query_endpoint_knowledge()