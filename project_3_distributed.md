# Project 3: Distributed Training with ZeRO

Train a 125M-parameter LLM with ZeRO Stage 2.

## Objective
Use Accelerate for distributed training on AWS SageMaker.

## Steps
1. Set up SageMaker with GPU instances.
2. Configure ZeRO Stage 2.
3. Train on Wikitext.
4. Save checkpoints to S3.

## Dataset
Wikitext (`huggingface/datasets/wikitext`).

## Run
```bash
python src/section_3_scaling/distributed.py
```
