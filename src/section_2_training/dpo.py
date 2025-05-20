import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

def dpo_loss(model, tokenizer, prompt, chosen_response, rejected_response, beta=0.1):
    # Tokenize inputs
    chosen_inputs = tokenizer(prompt + chosen_response, return_tensors="pt", truncation=True, padding=True)
    rejected_inputs = tokenizer(prompt + rejected_response, return_tensors="pt", truncation=True, padding=True)
    
    # Compute log probabilities
    chosen_logits = model(**chosen_inputs).logits
    rejected_logits = model(**rejected_inputs).logits
    
    chosen_log_prob = F.log_softmax(chosen_logits, dim=-1).mean()
    rejected_log_prob = F.log_softmax(rejected_logits, dim=-1).mean()
    
    # DPO loss
    loss = -F.logsigmoid(beta * (chosen_log_prob - rejected_log_prob))
    return loss

def train_dpo(model_name, preference_data, epochs=3):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch in preference_data:
            prompt, chosen_response, rejected_response = batch
            
            optimizer.zero_grad()
            loss = dpo_loss(model, tokenizer, prompt, chosen_response, rejected_response)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(preference_data)}")
    
    return model

if __name__ == "__main__":
    # Example preference data (placeholder)
    preference_data = [
        ("What is the capital of France?", "The capital is Paris.", "I think it's Florida."),
        # Add more data...
    ]
    model = train_dpo("distilgpt2", preference_data)
