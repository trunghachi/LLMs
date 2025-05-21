import pytest
from src.section_4_finetuning.lora import apply_lora, train_lora

def test_lora():
    # Sample dataset
    dataset = [
        ("What is 1+1?", "1+1 is 2."),
    ]
    
    # Apply LoRA
    model, tokenizer = apply_lora("distilgpt2")
    
    # Fine-tune with LoRA
    model = train_lora(model, tokenizer, dataset, epochs=1)
    
    # Check if model is returned
    assert model is not None, "LoRA fine-tuning failed to return a model"
