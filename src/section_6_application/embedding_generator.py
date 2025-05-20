from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
    
    def generate_embeddings(self, documents):
        # Generate embeddings for a list of documents
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        return embeddings
    
    def build_index(self, documents):
        # Generate embeddings
        embeddings = self.generate_embeddings(documents)
        
        # Create FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        
        return embeddings
    
    def search(self, query, k=5):
        # Generate embedding for the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search for top-k nearest documents
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

if __name__ == "__main__":
    # Example usage
    documents = [
        "The capital of France is Paris.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
    ]
    generator = EmbeddingGenerator()
    embeddings = generator.build_index(documents)
    distances, indices = generator.search("What is the capital of France?", k=2)
    print("Top documents:", [documents[i] for i in indices[0]])
