import pytest
from src.section_2_training.dpo import train_dpo

def test_dpo():
    # Sample preference data
    preference_data = [
        ("What is 1+1?", "1+1 is 2.", "1+1 is 11."),
    ]
    
    # Run DPO training
    model = train_dpo("distilgpt2", preference_data, epochs=1)
    
    # Check if model is returned
    assert model is not None, "DPO training failed to return a model"
