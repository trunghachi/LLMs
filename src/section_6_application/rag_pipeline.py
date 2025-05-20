from fastapi import FastAPI
from .embedding_generator import EmbeddingGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

class RAGPipeline:
    def __init__(self, model_name="distilgpt2", documents=None):
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        self.documents = documents
        if documents:
            self.embedding_generator.build_index(documents)
        
        # Initialize LLM
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def retrieve(self, query, k=3):
        # Retrieve top-k documents
        distances, indices = self.embedding_generator.search(query, k)
        retrieved_docs = [self.documents[i] for i in indices[0]]
        return retrieved_docs
    
    def generate(self, query, retrieved_docs):
        # Combine query and retrieved documents into a prompt
        context = "\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=100)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Initialize RAG pipeline with sample documents
documents = [
    "The capital of France is Paris.",
    "France is a country in Europe.",
    "The Eiffel Tower is in Paris.",
]
rag_pipeline = RAGPipeline(documents=documents)

@app.get("/rag")
async def rag_query(query: str):
    # Retrieve and generate
    retrieved_docs = rag_pipeline.retrieve(query)
    response = rag_pipeline.generate(query, retrieved_docs)
    return {"query": query, "response": response, "retrieved_docs": retrieved_docs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
