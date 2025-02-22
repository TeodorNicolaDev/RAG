import fitz  # PyMuPDF

################################################################################
# Function: extract_text_from_pdf
################################################################################
def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file.

    :param pdf_file: A file-like object representing the PDF.
    :return: A string with the extracted text.
    """
    try:
        # Use pdf_file.file.read() to synchronously read the file bytes
        doc = fitz.open(stream=pdf_file.file.read(), filetype="pdf")
        full_text = []
        for page in doc:
            page_text = page.get_text("text")
            full_text.append(page_text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""
    
################################################################################
# Function: chunk_text
################################################################################
    
def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits the extracted text into chunks of a specified word count.
    
    :param text: The full text extracted from the PDF.
    :param chunk_size: Number of words per chunk (default is 300).
    :param overlap: Number of words to overlap between chunks (default is 50).
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