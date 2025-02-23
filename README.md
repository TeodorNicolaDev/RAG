RAG Pipeline with Mistral

This repository implements a Retrieval-Augmented Generation (RAG) pipeline built from scratch using FastAPI, Mistral’s APIs, and Faiss. The system allows users to upload PDF files to create a knowledge base, and then answer queries by retrieving relevant information from those PDFs and generating answers using a language model.

The project uses the following technologies:

FastAPI – for building the backend API.
Uvicorn – an ASGI server to run the FastAPI application.
PyMuPDF (fitz) – for extracting text from PDF files.
Mistral AI API (mistralai) – for generating text embeddings and chat completions.
Faiss (faiss-cpu) – for efficient similarity search in high-dimensional vector spaces.
NumPy – for numerical operations and handling embedding vectors.
Requests – for making HTTP calls.
Scikit-learn – for computing cosine similarity.
The complete requirements.txt includes: fastapi uvicorn PyMuPDF mistralai faiss-cpu numpy requests scikit-learn

Installation
a. Clone the repository: git clone https://github.com/yourusername/your-repo.git cd your-repo

b. Create and activate a virtual environment: python -m venv venv For macOS/Linux: source venv/bin/activate For Windows: venv\Scripts\activate

c. Install dependencies: pip install -r requirements.txt

d. Set your Mistral API key: export MISTRAL_API_KEY="your_actual_api_key_here"

Running the Project
a. Launch the FastAPI app from the repository root: uvicorn rag_pipeline.app:app --reload The --reload flag enables automatic reloading during development.

b. Open your browser and navigate to: http://127.0.0.1:8000/docs Use the Swagger UI to test the endpoints:

/ingest: Upload PDF files.
/query: Send questions and receive generated answers.
Functionalities and Implementation Details
a. Data Ingestion:

Endpoint: /ingest
Functionality: Users upload PDFs which are processed by extracting text (using PyMuPDF), splitting the text into chunks, generating 1024-dimensional embeddings for each chunk via Mistral’s embeddings API, and storing both the original text and embeddings in global lists (TEXT_CHUNKS and EMBEDDINGS_LIST).
b. Text Chunking:

Strategy: The document text is split using paragraph boundaries (by splitting on double newlines). Paragraphs are then combined until a maximum length (default 2048 characters) is reached, with an overlap (default 100 characters) between chunks.
Rationale: This method preserves semantic boundaries and improves retrieval quality by ensuring important details (such as author names) are not lost.
c. Embedding Generation:

Approach: Each text chunk is sent to Mistral’s embeddings API (using the model "mistral-embed") to obtain a 1024-dimensional vector.
Rationale: These embeddings capture the semantic meaning of the text, enabling effective similarity search even if the language model’s training data is outdated.
d. Retrieval and Re-Ranking:

Process:
A Faiss index is built from the stored embeddings (EMBEDDINGS_LIST) to quickly retrieve a candidate set (default 10) based on L2 distance.
Each candidate is re-ranked using a weighted combination of cosine similarity (semantic match) and a keyword matching score (word overlap between the query and the chunk text). The combined score is computed as: combined_score = alpha * semantic_similarity + (1 - alpha) * keyword_score (with alpha = 0.7 by default).
The top 5 candidates (default) are selected and merged to form a coherent context.
Rationale: The two-stage retrieval process improves precision and ensures that key details are captured regardless of their position in the document.
e. Query Processing and Answer Generation:

Endpoint: /query
Flow:
Trivial Query Detection: The system first checks if a query is trivial (e.g., "hello") using a simple keyword-based approach. If trivial, it bypasses retrieval and directly generates a response.
For Knowledge Queries: The query is transformed and embedded. Relevant text chunks are retrieved from the knowledge base, merged into a context, and then used to construct a prompt.
Answer Generation: The prompt (which combines the retrieved context and the original query) is sent to Mistral’s chat completion API (using the model "mistral-large-latest") to generate a final answer.
Rationale: This design provides context-aware answers by ensuring that the language model leverages the most up-to-date document information when needed.
Testing
The project includes a comprehensive test suite:

test_mistral_embedding.py: Tests that the Mistral embeddings function returns a 1024-dimensional vector.
test_ingestion.py: Verifies that PDF ingestion, text extraction, improved chunking, and embedding storage work correctly.
test_query.py: Checks that the /query endpoint returns the expected structure and content for both trivial and knowledge-based queries.
To run all tests, execute: pytest To run a specific test, for example: pytest rag_pipeline/test_query.py::test_query_endpoint_knowledge

Potential Improvements
a. Trivial Query Detection:

Current Approach: Simple keyword-based check (e.g., "hello", "hi").
Future Improvements: Implement advanced NLP-based query classification to better decide when to retrieve external context versus using the model's internal knowledge.
b. Chunking and Retrieval:

Chunking: Experiment with sentence-based or token-based splitting (using NLP libraries like spaCy) to better preserve semantic units.
Retrieval: Enhance re-ranking by incorporating additional scoring methods (e.g., TF-IDF, BM25) or using more advanced vector search algorithms.
Overlap: Optimize overlap parameters to avoid redundancy while ensuring key information is captured.
c. Embedding Model:

Current Choice: Mistral’s embeddings.
Future Alternatives: Evaluate other embedding providers (such as Cohere or OpenAI) to potentially improve performance.
d. User Interface and Experience:

Develop a dedicated front-end interface for a better user experience.
Improve error handling and user feedback in the API.
e. Advanced Query Handling:

Consider techniques like Hypothetical Document Embeddings (HyDE) to generate better query embeddings.
Further differentiate queries that require external context from those that do not.
Conclusion
This RAG pipeline project provides a fully functional system for building a dynamic knowledge base from PDFs and generating context-aware answers. The project leverages FastAPI, Mistral’s APIs, and Faiss to deliver robust retrieval and generation. Future improvements include enhanced query classification, refined chunking and retrieval strategies, exploration of alternative embedding models, and a more polished user interface.