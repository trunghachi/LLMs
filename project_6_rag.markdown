# Project 6: RAG Application with FastAPI

Build a RAG-based Q&A system.

## Objective
Create a scalable Q&A system with FAISS and FastAPI.

## Steps
1. Index SQuAD dataset with Sentence-Transformers.
2. Implement retrieval with FAISS.
3. Generate answers with DistilGPT-2.
4. Deploy with FastAPI.

## Dataset
SQuAD (`huggingface/datasets/squad`).

## Run
```bash
python src/section_6_application/rag_pipeline.py
```