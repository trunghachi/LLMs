# Project 4: Fine-Tuning with QLoRA

Fine-tune LLaMA-7B with QLoRA for 4096-token context.

## Objective
Increase context length using QLoRA.

## Steps
1. Load LLaMA-7B with PEFT.
2. Fine-tune on a long-context dataset.
3. Evaluate perplexity.

## Dataset
Custom long-context dataset (e.g., PG-19).

## Run
```bash
python src/section_4_finetuning/qlora.py
```
