import pytest
from src.section_2_training.rlhf import train_rlhf

def test_rlhf():
    # Sample preference data
    preference_data = [
        ("What is 1+1?", "1+1 is 2.", "1+1 is 11.", 1),
    ]
    
    # Run RLHF training
    model = train_rlhf("distilgpt2", preference_data, epochs=1)
    
    # Check if model is returned
    assert model is not None, "RLHF training failed to return a model"
