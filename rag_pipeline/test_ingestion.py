import time
import tempfile
import fitz  # PyMuPDF
from fastapi.testclient import TestClient
from app import app
from search import EMBEDDINGS_DB

client = TestClient(app)

def create_sample_pdf():
    """
    Creates a temporary PDF with simple text.
    Returns the file path.
    """
    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc = fitz.open()  # Create a new PDF in memory
    page = doc.new_page()
    # Write a simple sentence
    page.insert_text((72, 72), "This is a test PDF file for semantic search ingestion.")
    doc.save(temp_pdf.name)
    doc.close()
    return temp_pdf.name

def test_ingest_endpoint():
    # Clear the global embeddings store before running the test.
    EMBEDDINGS_DB.clear()
    
    sample_pdf_path = create_sample_pdf()
    print(f"Created sample PDF at: {sample_pdf_path}")
    
    # Pause to avoid hitting rate limits.
    time.sleep(2)
    
    with open(sample_pdf_path, "rb") as pdf_file:
        response = client.post("/ingest", files=[("files", ("test.pdf", pdf_file, "application/pdf"))])
    
    print("Response from /ingest endpoint:")
    print(response.text)
    
    assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"
    data = response.json()
    
    print("Parsed JSON response:")
    print(data)
    
    assert "total_chunks" in data, "Response JSON missing 'total_chunks' key."
    assert data["total_chunks"] > 0, "Expected at least one chunk to be processed."
    
    print("Current state of EMBEDDINGS_DB:")
    print(EMBEDDINGS_DB)
    assert len(EMBEDDINGS_DB) == data["total_chunks"], "Global EMBEDDINGS_DB should match total chunks processed."