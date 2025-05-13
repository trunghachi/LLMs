# Section 1: The Transformer Architecture

This section explains the Transformer architecture and implements it in PyTorch.

## Introduction
Transformers, introduced in *Attention is All You Need* (2017), outperform RNNs for sequence modeling due to parallelization and attention mechanisms.

## Overall Architecture
Transformers consist of encoder and decoder stacks, with variants like GPT (decoder-only) and BERT (encoder-only).

## Components
- **Self-Attention**: Weighs token importance using queries (Q), keys (K), and values (V).
- **Multihead Attention**: Captures multiple relationships with parallel attention heads.
- **Position Embeddings**: Encodes sequence order (sinusoidal or learned).
- **Encoder**: Processes input sequences with self-attention and feed-forward networks.
- **Decoder**: Generates output sequences with masked self-attention.
- **Feed-Forward Networks**: Apply pointwise transformations.
- **Layer Normalization**: Stabilizes training.

## Implementation
Below is a PyTorch implementation of the self-attention layer (see `src/section_1_transformer/self_attention.py`).

```python
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_linear(context), attn_weights

# Test
def test_self_attention():
    d_model, n_heads, seq_len, batch_size = 64, 8, 10, 2
    model = SelfAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    output, attn_weights = model(x)
    print(f"Output shape: {output.shape}")
    plt.imshow(attn_weights[0, 0].detach().numpy(), cmap='viridis')
    plt.savefig("attention_weights.png")
```

Run with:
```bash
python src/section_1_transformer/self_attention.py
```

## Testing
- Verify output shapes and attention weights.
- Visualize attention with Matplotlib.

## Project
See [Project 1: Modern Transformer Implementation](project_1_transformer.md).
