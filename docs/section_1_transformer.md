# Section 1: The Transformer Architecture

This section explains the Transformer architecture and implements it in PyTorch.
The Transformer is the foundational architecture behind most modern Large Language Models (LLMs), revolutionizing natural language processing (NLP) with its ability to handle long-range dependencies and parallelize training. Introduced in the seminal paper *"Attention is All You Need"* by Vaswani et al. (2017), the Transformer has become the backbone of models like BERT, GPT, and beyond. In this section, you’ll explore the architecture in detail, understand its key components, and implement a simplified version from scratch using PyTorch.

## Objectives
- Understand the core concepts of the Transformer architecture.
- Learn how self-attention and multi-head attention enable efficient sequence modeling.
- Implement a basic Transformer model as a hands-on exercise.

## Background
The Transformer departs from traditional recurrent neural networks (RNNs) and convolutional neural networks (CNNs) by relying entirely on attention mechanisms. This allows it to:
- Process input sequences in parallel, significantly speeding up training.
- Capture long-range dependencies without the vanishing gradient problem common in RNNs.
- Scale effectively to large datasets and models.

The architecture consists of an encoder-decoder structure, with each component built from stacked layers of self-attention and feed-forward networks.

## Key Components

- **Self-Attention**: Weighs token importance using queries (Q), keys (K), and values (V).
- **Multihead Attention**: Captures multiple relationships with parallel attention heads.
- **Position Embeddings**: Encodes sequence order (sinusoidal or learned).
- **Encoder**: Processes input sequences with self-attention and feed-forward networks.
- **Decoder**: Generates output sequences with masked self-attention.
- **Feed-Forward Networks**: Apply pointwise transformations.
- **Layer Normalization**: Stabilizes training.

### 1. Self-Attention Mechanism
- **Purpose**: Allows the model to weigh the importance of different words in a sequence when encoding a particular word.
- **How It Works**:
  - Each token is transformed into query (Q), key (K), and value (V) vectors using learned linear transformations.
  - Attention scores are computed as `Attention(Q, K, V) = softmax((Q * K^T) / √d_k) * V`, where `d_k` is the dimension of the key vectors.
  - This mechanism enables the model to focus on relevant parts of the input.
- **Implementation**: See `src/section_1_transformer/self_attention.py` for a PyTorch implementation.

### 2. Multi-Head Attention
- **Purpose**: Extends self-attention by allowing the model to focus on different representation subspaces simultaneously.
- **How It Works**:
  - Splits the Q, K, and V vectors into multiple "heads," computes attention separately for each head, and concatenates the results.
  - This enhances the model’s ability to capture diverse relationships in the data.
- **Benefit**: Improves expressiveness and performance.

### 3. Feed-Forward Networks (FFN)
- **Purpose**: Applies a position-wise transformation to each token’s representation.
- **How It Works**:
  - Consists of two linear layers with a non-linear activation (e.g., ReLU) between them.
  - Applied independently to each position, adding non-linearity to the model.

### 4. Positional Encoding
- **Purpose**: Injects information about the order of tokens, since Transformers lack inherent sequence awareness.
- **How It Works**:
  - Adds sinusoidal or learned positional encodings to the input embeddings.
  - Ensures the model can distinguish between the position of words (e.g., "cat" at position 1 vs. position 5).

### 5. Encoder and Decoder Stacks
- **Encoder**: Processes the input sequence, producing a set of continuous representations.
- **Decoder**: Generates the output sequence, attending to both the encoder output and previously generated tokens (with masking for auto-regressive behavior).
- **Layers**: Typically, 6 layers are stacked, though this can scale with model size.

### 6. Layer Normalization and Residual Connections
- **Purpose**: Stabilize and accelerate training.
- **How It Works**:
  - Residual connections (skip connections) add the input of a sub-layer to its output.
  - Layer normalization is applied to normalize the outputs.

## Practical Implementation
In this section, you’ll implement a simplified Transformer model. The code is located in `src/section_1_transformer/`, with the following structure:
- `self_attention.py`: Implements the self-attention mechanism (see the provided code).
- `transformer.py`: Contains the full Transformer architecture (to be completed as part of the project).

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

### Project 1: Modern Transformer Implementation
- **Goal**: Build a working Transformer model from scratch.
- **Steps**:
  1. Extend `self_attention.py` to include multi-head attention.
  2. Implement the feed-forward network and positional encoding in `transformer.py`.
  3. Stack encoder and decoder layers to create a complete model.
  4. Test the model on a small dataset (e.g., a subset of Wikitext).
- **Details**: Refer to [Project 1: Modern Transformer Implementation](project_1_transformer.md) for a step-by-step guide, including code snippets and expected outputs.

## Next Steps
Once you’ve implemented the Transformer, proceed to [Section 2: Training LLMs to Follow Instructions](section_2_training.md) to learn how to train and align your model.

## Additional Resources
- Original Transformer Paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- PyTorch Documentation: [torch.nn](https://pytorch.org/docs/stable/nn.html)
- Hugging Face Transformers: [Transformers Library](https://huggingface.co/docs/transformers)
