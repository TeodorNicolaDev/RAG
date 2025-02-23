## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Running the Project](#running-the-project)
4. [Functionalities and Implementation Details](#functionalities-and-implementation-details)
   - [Data Ingestion](#data-ingestion)
   - [Text Chunking](#text-chunking)
   - [Embedding Generation](#embedding-generation)
   - [Retrieval and Re-Ranking](#retrieval-and-re-ranking)
   - [Query Processing and Answer Generation](#query-processing-and-answer-generation)
5. [Testing](#testing)
6. [Potential Improvements](#potential-improvements)
7. [Conclusion](#conclusion)

## Requirements

The project uses the following technologies:

- **FastAPI** – For building the backend API.
- **Uvicorn** – An ASGI server to run the FastAPI application.
- **PyMuPDF (fitz)** – For extracting text from PDF files.
- **Mistral AI API (mistralai)** – For generating text embeddings and chat completions.
- **Faiss (faiss-cpu)** – For efficient similarity search in high-dimensional vector spaces.
- **NumPy** – For numerical operations and handling embedding vectors.
- **Requests** – For making HTTP calls.
- **Scikit-learn** – For computing cosine similarity.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Create and activate a virtual environment:
 ```bash
python -m venv venv
source venv/bin/activate
```

3. *** Install dependencies:

 ```bash
pip install -r requirements.txt
 ```


4. *** Set your Mistral API key: Ensure that the environment variable MISTRAL_API_KEY is set:

 ```bash
export MISTRAL_API_KEY="your_actual_api_key_here"
 ```


## Running the Project

1. **Launch the FastAPI app:**
   From the repository root, run:
   ```bash
   uvicorn rag_pipeline.app:app --reload


## Functionalities and Implementation Details

### Data Ingestion

*   **Endpoint:** `/ingest`
*   **Functionality:**  
    Users upload PDF files, which are processed as follows:
    *   **Text Extraction:**  
        PDFs are read using PyMuPDF.
    *   **Chunking:**  
        The text is split into chunks using an improved strategy that:
        *   Splits the text by paragraph boundaries.
        *   Combines paragraphs until a maximum length (default 2048 characters) is reached.
        *   Adds an overlap (default 100 characters) between chunks to preserve key details.
    *   **Embedding Generation:**  
        Each chunk is sent to Mistral’s embeddings API (model: `mistral-embed`) to produce a 1024-dimensional vector.
    *   **Storage:**  
        The original text chunks and their embeddings are stored in global lists (`TEXT_CHUNKS` and `EMBEDDINGS_LIST`).

### Text Chunking

*   **Strategy:**
    *   **Paragraph Splitting:**  
        The document text is split based on paragraph boundaries (using double newlines `\n\n`).
    *   **Chunk Combining:**  
        Paragraphs are concatenated until the combined text reaches a maximum character count.
    *   **Overlap:**  
        A fixed number of characters (e.g., 100) are overlapped between consecutive chunks to preserve continuity.
*   **Rationale:**  
    This approach preserves semantic boundaries and improves retrieval quality by ensuring that important details (such as author names) are not lost.

### Embedding Generation

*   **Approach:**  
    Each text chunk is sent to Mistral’s embeddings API to generate a 1024-dimensional embedding.
    
*   **Rationale:**  
    The embeddings capture the semantic meaning of the text, enabling effective similarity search even if the language model’s training data is outdated.
    

### Retrieval and Re-Ranking

*   **Process:**
    1.  **Candidate Retrieval:**  
        A Faiss index is built from the stored embeddings (`EMBEDDINGS_LIST`) and used to retrieve a candidate set (default 10) based on L2 distance.
    2.  **Re-Ranking:**  
        Each candidate is re-ranked using a weighted combination of:
        
        *   **Cosine Similarity:**  
            Measures semantic similarity between the query embedding and the chunk embedding.
        *   **Keyword Matching Score:**  
            A simple score based on word overlap between the query and the chunk text.
        
        The combined score is computed as:
        `combined_score = alpha * semantic_similarity + (1 - alpha) * keyword_score`
        
        with a default alpha of 0.7.
    3.  **Final Selection:**  
        The top 5 candidates (default) are selected and merged to form the final context.
*   **Rationale:**  
    The two-stage retrieval process improves precision and ensures that key details are captured regardless of their position in the document.

### Query Processing and Answer Generation

*   **Endpoint:** `/query`
*   **Flow:**
    1.  **Trivial Query Detection:**  
        The system checks if a query is trivial (e.g., "hello"). If so, it bypasses retrieval and directly generates a simple response.
    2.  **Knowledge Queries:**
        *   The query is transformed (normalized) and embedded.
        *   Relevant text chunks are retrieved from the knowledge base and merged into a coherent context.
        *   A prompt is constructed that includes the retrieved context and the original query.
        *   Mistral’s chat completion API (model: `mistral-large-latest`) is used to generate the final answer.
*   **Rationale:**  
    This design ensures that when external context is needed, the language model is provided with the most relevant, up-to-date information extracted from the uploaded PDFs.

## Testing

The project includes a comprehensive test suite:

*   **test\_mistral\_embedding.py:**  
    Validates that the embedding function returns a 1024-dimensional vector.
*   **test\_ingestion.py:**  
    Verifies that PDF ingestion, text extraction, improved chunking, and embedding storage work correctly.
*   **test\_query.py:**  
    Checks that the `/query` endpoint returns the expected response for both trivial queries (e.g., "hello") and knowledge-based queries (e.g., "Who are the authors of the paper?").

To run all tests, execute:

```bash
`pytest`
```

To run an individual test, for example:

```bash
`pytest rag_pipeline/test_query.py::test_query_endpoint_knowledge`
```

## Potential Improvements

1.  **Trivial Query Detection:**
    
    *   **Current Approach:**  
        Uses a simple keyword-based check (e.g., "hello", "hi").
    *   **Future Improvements:**  
        Implement advanced NLP-based classification to better decide when to use external retrieval versus the model's internal knowledge.
2.  **Chunking and Retrieval:**
    
    *   **Chunking:**  
        Explore sentence-based or token-based splitting using NLP libraries (e.g., spaCy) to further preserve semantic boundaries.
    *   **Retrieval:**  
        Enhance re-ranking by incorporating additional methods (e.g., TF-IDF, BM25) or more advanced nearest-neighbor search algorithms.
    *   **Overlap Optimization:**  
        Fine-tune the overlap parameter to balance redundancy and context preservation.
3.  **Embedding Model:**
    
    *   **Current Model:**  
        Mistral’s embeddings are used.
    *   **Alternatives:**  
        Evaluate other embedding providers (e.g., Cohere, OpenAI) to potentially improve performance.
4.  **User Interface and Experience:**
    
    *   Develop a dedicated front-end for a more user-friendly experience.
    *   Enhance error handling and provide clearer feedback to the user.
5.  **Advanced Query Handling:**
    
    *   Consider techniques such as Hypothetical Document Embeddings (HyDE) to improve query embeddings.
    *   Further differentiate between queries that rely on the model’s internal knowledge and those that require external knowledge retrieval.

## Conclusion

This RAG pipeline project provides a fully functional system for building a dynamic knowledge base from PDFs and generating context-aware answers. By leveraging FastAPI, Mistral’s APIs, and Faiss for vector search, the system effectively ingests documents, extracts and intelligently chunks text, and retrieves relevant context to augment answer generation. Future improvements will focus on advanced query classification, refined chunking and retrieval strategies, exploring alternative embedding models, and enhancing the overall user interface.










