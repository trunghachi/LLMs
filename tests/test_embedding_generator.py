import pytest
import numpy as np
from src.section_6_application.embedding_generator import EmbeddingGenerator

def test_embedding_generator():
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Sample documents
    documents = ["This is a test.", "Another test."]
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(documents)
    
    # Check embeddings
    assert embeddings.shape == (2, 384), "Embedding shape mismatch (expected 384 for all-MiniLM-L6-v2)"
    assert not np.isnan(embeddings).any(), "Embeddings contain NaN values"
    
    # Build index and search
    generator.build_index(documents)
    distances, indices = generator.search("Test query", k=1)
    
    # Check search results
    assert len(distances[0]) == 1, "Search returned wrong number of results"
    assert len(indices[0]) == 1, "Search returned wrong number of indices"
