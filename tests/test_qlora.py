import pytest
from src.section_4_finetuning.qlora import apply_qlora, train_qlora

def test_qlora():
    # Sample dataset
    dataset = [
        ("What is 1+1?", "1+1 is 2."),
    ]
    
    # Apply QLoRA
    model, tokenizer = apply_qlora("distilgpt2")
    
    # Fine-tune with QLoRA
    model = train_qlora(model, tokenizer, dataset, epochs=1)
    
    # Check if model is returned
    assert model is not None, "QLoRA fine-tuning failed to return a model"
