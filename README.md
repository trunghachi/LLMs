# Build Your Own Large Language Models From Scratch

This repository provides a comprehensive guide to building, training, fine-tuning, deploying, and applying Large Language Models (LLMs) for production use. It includes theoretical explanations, practical implementations in PyTorch, and hands-on projects.

## Target Audience
Intermediate to advanced machine learning engineers familiar with:
- Python and PyTorch.
- Linear algebra, probability, and neural networks.
- Basic ML concepts (e.g., supervised learning, optimization).

## Recap
```
LLMs/
├── data/
│   ├── pretraining/
│   ├── finetuning/
│   ├── rag/
├── docs/
│   ├── README.md
│   ├── section_0_introduction.md
│   ├── section_1_transformer.md
│   ├── section_2_training.md
│   ├── section_3_scaling.md
│   ├── section_4_finetuning.md
│   ├── section_5_deployment.md
│   ├── section_6_application.md
│   ├── section_7_ethics.md
│   ├── section_8_capstone.md
│   ├── project_1_transformer.md
│   ├── project_2_alignment.md
│   ├── project_3_distributed.md
│   ├── project_4_qlora.md
│   ├── project_5_deployment.md
│   ├── project_6_rag.md
│   ├── project_7_content_filter.md
├── src/
│   ├── section_1_transformer/
│   │   ├── self_attention.py
│   ├── section_2_training/
│   │   ├── rlhf.py
│   ├── section_4_finetuning/
│   │   ├── qlora.py
│   ├── section_6_application/
│   │   ├── rag_pipeline.py
├── tests/
├── requirements.txt
├── setup_repo.sh
```
## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/trunghachi/LLMs.git
   cd llm-from-scratch
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download datasets (e.g., Wikitext, Alpaca):
   ```bash
   python scripts/download_datasets.py
   ```

## Requirements
- Python 3.8+
- PyTorch, Transformers, TRL, PEFT, Accelerate, vLLM, FastAPI, FAISS, Sentence-Transformers, Matplotlib

## Document Structure
The guide is divided into eight sections, each with a hands-on project:

1. **[Introduction](docs/section_0_introduction.md)**: Overview, audience, and tools.
2. **[The Transformer Architecture](docs/section_1_transformer.md)**: Understand and implement Transformers.
   - [Project 1: Modern Transformer Implementation](docs/project_1_transformer.md)
3. **[Training LLMs to Follow Instructions](docs/section_2_training.md)**: Pretraining, fine-tuning, and alignment.
   - [Project 2: Aligning with RLHF and DPO](docs/project_2_alignment.md)
4. **[Scaling Model Training](docs/section_3_scaling.md)**: Distributed training and hardware optimization.
   - [Project 3: Distributed Training with ZeRO](docs/project_3_distributed.md)
5. **[Fine-Tuning LLMs](docs/section_4_finetuning.md)**: Efficient fine-tuning with LoRA/QLoRA.
   - [Project 4: Fine-Tuning with QLoRA](docs/project_4_qlora.md)
6. **[Deploying LLMs](docs/section_5_deployment.md)**: Scalable and secure deployment.
   - [Project 5: LLM Endpoint with vLLM](docs/project_5_deployment.md)
7. **[Building the Application Layer](docs/section_6_application.md)**: RAG-based applications.
   - [Project 6: RAG Application with FastAPI](docs/project_6_rag.md)
8. **[Ethical Considerations and Safety](docs/section_7_ethics.md)**: Bias, fairness, and safety.
   - [Project 7: Content Filter](docs/project_7_content_filter.md)
9. **[Capstone Project](docs/section_8_capstone.md)**: Build a complete LLM application.

## Learning Outcomes
- Implement a Transformer from scratch.
- Train and align LLMs with RLHF, DPO, and ORPO.
- Scale training with distributed systems.
- Fine-tune efficiently with LoRA/QLoRA.
- Deploy LLMs with optimized inference.
- Build production-ready RAG applications.
- Address ethical and safety concerns.

## Contributing
Contributions are welcome! Please submit pull requests or open issues.
