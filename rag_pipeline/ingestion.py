import fitz  # PyMuPDF
from search import get_embedding_from_mistral, EMBEDDINGS_DB

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file using PyMuPDF.
    
    :param pdf_file: A file-like object from FastAPI's UploadFile.
    :return: A string containing the full text of the PDF.
    """
    try:
        # Use the underlying file object (synchronously reading the bytes)
        doc = fitz.open(stream=pdf_file.file.read(), filetype="pdf")
        full_text = []
        for page in doc:
            page_text = page.get_text("text")
            full_text.append(page_text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits the extracted text into overlapping chunks.
    
    :param text: The text to split.
    :param chunk_size: The number of words per chunk.
    :param overlap: The number of words to overlap between consecutive chunks.
    :return: A list of text chunks.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    for start in range(0, len(words), chunk_size - overlap):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
    return chunks

def process_pdf_and_store_embeddings(pdf_file):
    """
    Processes a PDF by extracting text, chunking it, generating embeddings for each chunk using Mistral,
    and storing each chunk and its embedding in the global EMBEDDINGS_DB.
    
    :param pdf_file: The uploaded PDF file.
    :return: The number of chunks processed.
    """
    text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(text)
    count = 0
    for chunk in chunks:
        embedding = get_embedding_from_mistral(chunk)
        if embedding is not None:
            # Use the current length of EMBEDDINGS_DB as a unique ID.
            chunk_id = len(EMBEDDINGS_DB)
            EMBEDDINGS_DB[chunk_id] = {"text": chunk, "embedding": embedding}
            count += 1
    return count