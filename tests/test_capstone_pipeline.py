import pytest
from fastapi.testclient import TestClient
from src.capstone.capstone_pipeline import app, CapstonePipeline

client = TestClient(app)

def test_capstone_pipeline():
    # Initialize pipeline
    documents = ["The capital of France is Paris."]
    pipeline = CapstonePipeline(documents=documents)
    
    # Test retrieval and generation
    response = pipeline.generate("What is the capital of France?")
    assert response is not None, "Capstone pipeline failed to generate response"
    assert "Paris" in response, "Generated response does not contain expected content"

def test_capstone_endpoint():
    # Test API endpoint
    response = client.get("/qa", params={"query": "What is the capital of France?"})
    assert response.status_code == 200, "Capstone endpoint failed"
    assert "response" in response.json(), "Response missing 'response' field"
    assert "Paris" in response.json()["response"], "Capstone response does not contain expected content"
