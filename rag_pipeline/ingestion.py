import fitz  # PyMuPDF
from rag_pipeline.search import get_embedding_from_mistral, add_chunk

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file.
    Supports both FastAPI UploadFile objects and standard file objects.
    """
    try:
        # Use pdf_file.file.read() if available, otherwise pdf_file.read()
        if hasattr(pdf_file, "file"):
            stream = pdf_file.file.read()
        else:
            stream = pdf_file.read()
        doc = fitz.open(stream=stream, filetype="pdf")
        full_text = []
        for page in doc:
            full_text.append(page.get_text("text"))
        return "\n\n".join(full_text)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def improved_chunk_text(text, max_chunk_length=2048, overlap=100):
    """
    Splits text into chunks using paragraph boundaries.
    Combines paragraphs until reaching max_chunk_length, and includes an overlap.
    """
    paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        candidate = (current_chunk + "\n\n" + para) if current_chunk else para
        if len(candidate) <= max_chunk_length:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap to preserve context.
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                chunks.append(para)
                current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def process_pdf_and_store_embeddings(pdf_file):
    """
    Processes a PDF: extracts text, splits it into chunks using improved_chunk_text,
    generates embeddings for each chunk, and stores them.
    
    Returns the number of chunks processed.
    """
    text = extract_text_from_pdf(pdf_file)
    chunks = improved_chunk_text(text)
    count = 0
    for chunk in chunks:
        embedding = get_embedding_from_mistral(chunk)
        if embedding is not None:
            add_chunk(chunk, embedding)
            count += 1
    return count