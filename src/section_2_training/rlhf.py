import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
import numpy as np

class RewardModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super(RewardModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.fc = nn.Linear(self.model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, -1, :]  # Last token hidden state
        reward = self.fc(hidden_states)
        return reward

def train_rlhf(model_name, preference_data, epochs=3):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize reward model
    reward_model = RewardModel(model_name)
    
    # PPO configuration
    config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=16,
    )
    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
    )
    
    # Training loop
    for epoch in range(epochs):
        for batch in preference_data:  # Assume preference_data is a list of (prompt, response1, response2, preference)
            prompt, response1, response2, preference = batch
            
            # Tokenize inputs
            prompt_ids = tokenizer(prompt, return_tensors="pt")
            response1_ids = tokenizer(response1, return_tensors="pt")
            response2_ids = tokenizer(response2, return_tensors="pt")
            
            # Compute rewards
            reward1 = reward_model(**response1_ids).mean()
            reward2 = reward_model(**response2_ids).mean()
            
            # PPO update
            query_tensors = prompt_ids["input_ids"]
            response_tensors = [response1_ids["input_ids"], response2_ids["input_ids"]]
            rewards = [reward1, reward2]
            
            ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        print(f"Epoch {epoch + 1}/{epochs} completed.")
    
    return model

if __name__ == "__main__":
    # Example preference data (placeholder)
    preference_data = [
        ("What is the capital of France?", "The capital is Paris.", "I think it's Florida.", 1),
        # Add more data...
    ]
    model = train_rlhf("distilgpt2", preference_data)
