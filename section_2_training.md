# Section 2: Training LLMs to Follow Instructions

This section covers pretraining, supervised fine-tuning (SFT), and alignment with RLHF, DPO, and ORPO.

## Overview
LLM training involves:
- **Pretraining**: Learn language patterns.
- **SFT**: Adapt to instructions.
- **Alignment**: Align with human preferences.

## Components
- **Causal Language Modeling**: Next-token prediction (e.g., CommonCrawl).
- **SFT**: Instruction tuning (e.g., Alpaca).
- **RLHF/DPO/ORPO**: Use human feedback or preferences.
- **Datasets**: Preprocessing and tokenization.
- **Evaluation**: Perplexity, BLEU, ROUGE, human scores.

## Implementation
Use Hugging Face TRL for RLHF/DPO.

## Project
See [Project 2: Aligning with RLHF and DPO](project_2_alignment.md).
