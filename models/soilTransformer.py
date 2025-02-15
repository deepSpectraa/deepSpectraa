import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    """ Multi-Head Self-Attention mechanism for feature extraction. """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads."

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear projection & reshape for multi-head attention
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (query, key, value) -> (batch, heads, seq_len, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)  # Scaled Dot-Product Attention
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # Apply attention weights

        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)  # Reshape to original
        return self.proj(out)


class FeedForward(nn.Module):
    """ Fully connected feedforward network with dropout. """
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """ A single Transformer encoder block with attention and feedforward layers. """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)  # Residual Connection
        ff_output = self.ff(x)
        return self.norm2(x + ff_output)  # Residual Connection


class soilTransformerModel(nn.Module):
    """ Transformer-based Model for Soil Property Prediction. """
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.1):
        super().__init__()

        # Initial Projection Layer
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Transformer Layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output Layer
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x.unsqueeze(1))  # Add sequence dimension
        for transformer in self.transformer_blocks:
            x = transformer(x)
        return self.output_layer(x.squeeze(1))  # Remove sequence dimension


# Ensure model is importable
__all__ = ["soilTransformerModel"]
