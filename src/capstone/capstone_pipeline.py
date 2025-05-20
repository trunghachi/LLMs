from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.section_4_finetuning.qlora import apply_qlora, train_qlora
from src.section_6_application.embedding_generator import EmbeddingGenerator
from src.section_7_ethics.content_filter import ContentFilter

app = FastAPI()

class CapstonePipeline:
    def __init__(self, model_name="distilgpt2", documents=None):
        # Initialize model and tokenizer
        self.model, self.tokenizer = apply_qlora(model_name)  # Apply QLoRA
        
        # Fine-tune the model (simplified)
        dataset = [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("What is the Eiffel Tower?", "The Eiffel Tower is a landmark in Paris."),
        ]
        self.model = train_qlora(self.model, self.tokenizer, dataset, epochs=1)
        
        # Initialize RAG components
        self.embedding_generator = EmbeddingGenerator()
        self.documents = documents or [
            "The capital of France is Paris.",
            "The Eiffel Tower is a landmark in Paris.",
        ]
        self.embedding_generator.build_index(self.documents)
        
        # Initialize content filter
        self.content_filter = ContentFilter()
    
    def retrieve(self, query, k=2):
        distances, indices = self.embedding_generator.search(query, k)
        return [self.documents[i] for i in indices[0]]
    
    def generate(self, query):
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)
        context = "\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=50)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Apply content filter
        if self.content_filter.is_safe(response):
            return response
        return "Content blocked due to potential toxicity."

# Initialize pipeline
pipeline = CapstonePipeline()

@app.get("/qa")
async def question_answering(query: str):
    response = pipeline.generate(query)
    return {"query": query, "response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
