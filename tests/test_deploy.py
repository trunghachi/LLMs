import pytest
from fastapi.testclient import TestClient
from src.section_5_deployment.deploy import app

client = TestClient(app)

def test_deploy_endpoint():
    # Test API endpoint
    response = client.get("/generate", params={"prompt": "Hello"})
    assert response.status_code == 200, "Deployment endpoint failed"
    assert "response" in response.json(), "Response missing 'response' field"
    assert response.json()["response"], "Generated response is empty"
