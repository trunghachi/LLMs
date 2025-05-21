import pytest
import torch
from src.section_1_transformer.self_attention import SelfAttention

def test_self_attention():
    # Setup
    embed_size = 64
    heads = 8
    batch_size = 2
    seq_len = 10
    
    # Initialize self-attention
    attention = SelfAttention(embed_size=embed_size, heads=heads)
    
    # Create dummy input
    x = torch.rand(batch_size, seq_len, embed_size)
    
    # Test forward pass
    output = attention(x, x, x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, embed_size), "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN values"
