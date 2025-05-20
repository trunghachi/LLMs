import torch
import torch.nn as nn
import math
from .self_attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        # Add & Norm
        x = self.norm1(attention + query)
        x = self.dropout(x)
        # Feed-forward
        forward = self.feed_forward(x)
        # Add & Norm
        out = self.norm2(forward + x)
        out = self.dropout(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_size)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        embed_size=256,
        num_layers=6,
        heads=8,
        forward_expansion=4,
        dropout=0.1,
        max_len=5000,
    ):
        super(Transformer, self).__init__()
        
        self.embed_size = embed_size
        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        
        # Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, src_len)
        return src_mask
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # Embeddings and positional encoding
        src_embedded = self.dropout(self.positional_encoding(self.src_word_embedding(src)))
        trg_embedded = self.dropout(self.positional_encoding(self.trg_word_embedding(trg)))
        
        # Encoder
        enc_out = src_embedded
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, enc_out, enc_out, src_mask)
        
        # Decoder
        dec_out = trg_embedded
        for layer in self.decoder_layers:
            dec_out = layer(enc_out, enc_out, dec_out, trg_mask)
        
        output = self.fc_out(dec_out)
        return output
