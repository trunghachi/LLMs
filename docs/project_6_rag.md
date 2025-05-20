# Project 6: RAG Application with FastAPI

This project focuses on building a Retrieval-Augmented Generation (RAG) application for question answering, using FAISS for retrieval, Sentence-Transformers for embeddings, and FastAPI for the API layer.

## Objectives
- Build a RAG pipeline for question answering.
- Expose the pipeline via a FastAPI endpoint.

## Steps
1. **Index Documents**:
   - Use a small document set (e.g., Wikipedia articles in `data/rag/`).
   - Generate embeddings with Sentence-Transformers.
   - Index embeddings with FAISS.
   - **Code Placeholder**: Implement indexing in `embedding_generator.py`.
2. **Build the RAG Pipeline**:
   - Retrieve relevant documents for a query.
   - Generate a response using the LLM.
   - **Code Placeholder**: Implement RAG in `rag_pipeline.py`.
3. **Expose via FastAPI**:
   - Create an endpoint for question answering.
   - **Code Placeholder**: Add FastAPI endpoint in `rag_pipeline.py`.
4. **Test the Application**:
   - Send sample questions and evaluate responses.

## Dataset
SQuAD (`huggingface/datasets/squad`).

## Run
```bash
python src/section_6_application/rag_pipeline.py
```
## Expected Output
- A working RAG-based question-answering system.
- Sample responses showing improved accuracy with retrieval.

## Next Steps
Proceed to [Section 7: Ethical Considerations and Safety](section_7_ethics.md).
