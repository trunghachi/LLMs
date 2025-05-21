import pytest
from fastapi.testclient import TestClient
from src.section_6_application.rag_pipeline import app, RAGPipeline

client = TestClient(app)

def test_rag_pipeline():
    # Initialize pipeline
    documents = ["The capital of France is Paris."]
    pipeline = RAGPipeline(documents=documents)
    
    # Test retrieval
    retrieved_docs = pipeline.retrieve("What is the capital of France?")
    assert len(retrieved_docs) > 0, "Retrieval failed to return documents"
    assert "Paris" in retrieved_docs[0], "Retrieved document does not contain expected content"
    
    # Test generation
    response = pipeline.generate("What is the capital of France?")
    assert response is not None, "Generation failed to return a response"
    assert "Paris" in response, "Generated response does not contain expected content"

def test_rag_endpoint():
    # Test API endpoint
    response = client.get("/rag", params={"query": "What is the capital of France?"})
    assert response.status_code == 200, "API endpoint failed"
    assert "response" in response.json(), "Response missing 'response' field"
    assert "Paris" in response.json()["response"], "API response does not contain expected content"
