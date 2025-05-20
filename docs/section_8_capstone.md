# Section 8: Capstone Project

The capstone project brings together everything you’ve learned in this guide to build a complete Large Language Model (LLM) application. You’ll design, train, fine-tune, deploy, and apply an LLM in a real-world scenario, addressing ethical concerns along the way.

## Objectives
- Integrate skills from all previous sections: Transformer, training, deployment, RAG, and ethics.
- Build a production-ready LLM application.
- Evaluate and improve the application.

## Project Overview
- **Task**: Build a question-answering system for a specific domain (e.g., medical Q&A).
- **Components**:
  - A Transformer-based LLM pretrained on a general corpus.
  - Fine-tuning on a domain-specific dataset with SFT/DPO, QLoRA.
  - Scale with ZeRO.
  - A RAG pipeline for retrieving relevant documents.
  - A deployed API endpoint with content filtering.
- **Steps**:
  1. Pretrain a small Transformer model (reuse code from Section 1).
  2. Fine-tune with QLoRA on a domain-specific dataset (Section 4).
  3. Build a RAG pipeline for retrieval (Section 6).
  4. Deploy the model with vLLM and FastAPI (Section 5).
  5. Add a content filter to ensure safe outputs (Section 7).

## Practical Implementation
The code will combine components from previous sections:
- `src/capstone/`: Contains the capstone project code.
  - **Code Placeholder**: Implement the full pipeline, integrating pretraining, fine-tuning, RAG, deployment, and content filtering.

## Deliverables
- A working question-answering system.
- Documentation of the process, challenges, and results.
- A demo showing the system in action.
- Web app with Gradio UI and FastAPI backend.

## Run
```bash
python src/capstone/app.py
```
## Next Steps
Congratulations on completing the guide! Share your project with the community by submitting a pull request.

## Additional Resources
- Full Guide Recap: [README.md](../README.md)
