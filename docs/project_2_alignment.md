# Project 2: Aligning with RLHF and DPO

Align a pretrained LLM using RLHF and DPO.

## Objective
Fine-tune GPT-2 with SFT and align with DPO using UltraFeedback.

## Steps
1. Fine-tune with Alpaca dataset.
2. Apply DPO with TRL.
3. Evaluate human preference scores.

## Dataset
- Alpaca (`huggingface/datasets/alpaca`).
- UltraFeedback (`huggingface/datasets/ultrafeedback`).

## Run
```bash
python src/section_2_training/dpo.py
```
