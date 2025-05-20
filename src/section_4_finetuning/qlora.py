import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def apply_qlora(model_name, lora_rank=8, lora_alpha=16):
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
    )
    
    # Apply LoRA to the quantized model
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def train_qlora(model, tokenizer, dataset, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataset:
            inputs = tokenizer(batch[0], return_tensors="pt", padding=True, truncation=True)
            targets = tokenizer(batch[1], return_tensors="pt", padding=True, truncation=True)
            
            outputs = model(**inputs, labels=targets["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataset)}")
    
    return model

if __name__ == "__main__":
    # Example dataset (placeholder)
    dataset = [
        ("What is the capital of France?", "The capital of France is Paris."),
        # Add more data...
    ]
    model, tokenizer = apply_qlora("distilgpt2")
    model = train_qlora(model, tokenizer, dataset)
