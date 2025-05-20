from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Simple deployment class (simulating vLLM behavior)
class LLMDeploy:
    def __init__(self, model_name="distilgpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def generate(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize deployment
llm_deploy = LLMDeploy()

@app.get("/generate")
async def generate_text(prompt: str):
    response = llm_deploy.generate(prompt)
    return {"prompt": prompt, "response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
