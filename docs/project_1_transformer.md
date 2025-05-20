# Project 1: Modern Transformer Implementation

This project guides you through implementing a Transformer model from scratch using PyTorch. You’ll build on the self-attention mechanism from Section 1 and create a complete model for a simple language modeling task.

## Objectives
- Implement a full Transformer architecture.
- Train the model on a small dataset.
- Evaluate the model’s performance.

## Steps
1. **Extend Self-Attention to Multi-Head Attention**:
   - Use `src/section_1_transformer/self_attention.py` as a starting point.
   - **Code Placeholder**: Add multi-head attention logic.
2. **Implement Positional Encoding**:
   - Add sinusoidal positional encodings to the input embeddings.
   - **Code Placeholder**: Implement positional encoding in `transformer.py`.
3. **Build the Transformer**:
   - Stack encoder and decoder layers with feed-forward networks.
   - **Code Placeholder**: Implement the full Transformer in `transformer.py`.
4. **Train on a Small Dataset**:
   - Use a subset of Wikitext (located in `data/pretraining/`).
   - Train for a few epochs and monitor loss.
   - **Code Placeholder**: Implement the training loop in `transformer.py`.
5. **Evaluate**:
   - Compute perplexity on a validation set.
   - **Code Placeholder**: Implement evaluation logic.

## Run
```bash
python src/section_1_transformer/transformer.py
```

## Expected Output
- A trained Transformer model with decreasing loss.
- Perplexity score on the validation set.

## Next Steps
Proceed to [Section 2: Training LLMs to Follow Instructions](section_2_training.md).
