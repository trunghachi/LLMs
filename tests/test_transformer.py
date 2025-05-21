import pytest
import torch
from src.section_1_transformer.transformer import Transformer

def test_transformer():
    # Setup
    src_vocab_size = 1000
    trg_vocab_size = 1000
    embed_size = 64
    num_layers = 2
    heads = 8
    
    # Initialize Transformer
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
    )
    
    # Create dummy input
    batch_size = 2
    src_seq_len = 10
    trg_seq_len = 10
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    trg = torch.randint(0, trg_vocab_size, (batch_size, trg_seq_len))
    
    # Test forward pass
    output = transformer(src, trg)
    
    # Check output shape
    assert output.shape == (batch_size, trg_seq_len, trg_vocab_size), "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN values"
