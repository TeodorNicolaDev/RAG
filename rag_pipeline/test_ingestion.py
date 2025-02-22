from fastapi.testclient import TestClient
from app import app
import tempfile
import fitz

client = TestClient(app)

def create_sample_pdf():
    """
    Create a temporary PDF file with a simple text content.
    """
    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc = fitz.open()  # create a new PDF in memory
    page = doc.new_page()
    page.insert_text((72, 72), "This is a test PDF file for ingestion.")
    doc.save(temp_pdf.name)
    doc.close()
    return temp_pdf.name

def test_ingest_endpoint():
    # Create a sample PDF file
    sample_pdf_path = create_sample_pdf()
    with open(sample_pdf_path, "rb") as pdf_file:
        response = client.post("/ingest", files=[("files", ("test.pdf", pdf_file, "application/pdf"))])
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert data["total_chunks"] > 0