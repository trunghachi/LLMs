# Section 6: Building the Application Layer

Develop production-ready LLM applications.

## Overview
Applications like chatbots and search systems.

## RAG Pipeline
- **Indexing**: Embed with Sentence-Transformers.
- **Retrieval**: FAISS for vector search.
- **Generation**: Pretrained LLM.

## Implementation
Below is a FastAPI-based RAG pipeline (see `src/section_6_application/rag_pipeline.py`).

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import uvicorn

app = FastAPI(title="RAG Pipeline")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline('text-generation', model='distilgpt2')
index = faiss.IndexFlatL2(384)
documents = []

class Document(BaseModel):
    text: str

class Query(BaseModel):
    text: str

@app.post("/index")
async def index_document(doc: Document):
    embedding = embedder.encode([doc.text])[0]
    index.add(np.array([embedding]).astype('float32'))
    documents.append(doc.text)
    return {"status": "Document indexed"}

@app.post("/query")
async def query_rag(query: Query):
    if not documents:
        raise HTTPException(status_code=400, detail="No documents indexed")
    query_embedding = embedder.encode([query.text])[0]
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k=2)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = " ".join(retrieved_docs)
    prompt = f"Question: {query.text}\nContext: {context}\nAnswer:"
    response = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return {"answer": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run with:
```bash
python src/section_6_application/rag_pipeline.py
```

## Project
See [Project 6: RAG Application with FastAPI](project_6_rag.md).