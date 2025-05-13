# Project 1: Modern Transformer Implementation

Build a GPT-like Transformer for text generation.

## Objective
Implement a decoder-only Transformer using PyTorch, integrating the self-attention layer from Section 1.

## Steps
1. Extend `SelfAttention` to `MultiheadAttention`.
2. Add position embeddings (sinusoidal).
3. Implement decoder blocks with layer normalization.
4. Train on a small dataset (e.g., Wikitext).
5. Generate text using top-k sampling.

## Code
See `src/section_1_transformer/transformer.py` for the full implementation.

## Dataset
Use Wikitext (`huggingface/datasets/wikitext`).

## Evaluation
- Measure perplexity.
- Visualize generated text.

## Run
```bash
python src/section_1_transformer/transformer.py
```