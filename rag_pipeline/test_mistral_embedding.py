import os
import numpy as np
from search import get_embedding_from_mistral

def test_get_embedding_from_mistral():
    # Ensure the API key is set in the environment for testing.
    os.environ["MISTRAL_API_KEY"] = "r6cJTcdQ93jseOPcR8jlSgWDC87PmREn"
    
    sample_text = "Test embedding generation using Mistral API."
    embedding = get_embedding_from_mistral(sample_text)
    print("Embedding:", embedding)
    
    assert embedding is not None, "No embedding returned."
    assert isinstance(embedding, np.ndarray), f"Expected numpy array, got {type(embedding)}."
    assert embedding.shape[0] == 1024, f"Expected embedding dimension 1024, got {embedding.shape[0]}."

if __name__ == "__main__":
    test_get_embedding_from_mistral()