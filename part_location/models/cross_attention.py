"""
Cross-Attention fusion module with spatial downsampling.

Implements spatial downsampling of patch tokens via AdaptiveAvgPool2d, then
multi-layer Cross-Attention where part tokens attend to whole tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDownsample(nn.Module):
    """
    Reshape patch tokens to 2D spatial grid and downsample via AdaptiveAvgPool2d.

    Input: [B, N, D] where N = H_p * W_p (e.g. 73*73 = 5329)
    Output: [B, N', D] where N' = target_size * target_size (e.g. 32*32 = 1024)
    """

    def __init__(self, target_size: int = 32):
        super().__init__()
        self.target_size = target_size
        self.pool = nn.AdaptiveAvgPool2d((target_size, target_size))

    def forward(self, x: torch.Tensor, grid_size: int) -> torch.Tensor:
        """
        Args:
            x: patch tokens [B, N, D]
            grid_size: spatial grid side length (e.g. 73 for 1024/14)

        Returns:
            downsampled tokens [B, target_size^2, D]
        """
        B, N, D = x.shape
        # Reshape to spatial grid: [B, D, H, W]
        x = x.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        # Downsample: [B, D, target_size, target_size]
        x = self.pool(x)
        # Flatten back: [B, target_size^2, D]
        x = x.flatten(2).transpose(1, 2)
        return x


class CrossAttentionLayer(nn.Module):
    """
    Single Cross-Attention layer.

    Query: part tokens, Key/Value: whole tokens.
    Uses F.scaled_dot_product_attention for memory-efficient computation.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = dropout

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: part tokens [B, Nq, D]
            key_value: whole tokens [B, Nkv, D]

        Returns:
            attended tokens [B, Nq, D]
        """
        B, Nq, D = query.shape
        Nkv = key_value.shape[1]

        # Pre-norm
        q = self.norm1(query)
        kv = self.norm2(key_value)

        # Project to Q, K, V
        Q = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).reshape(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).reshape(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)

        # Memory-efficient attention
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).reshape(B, Nq, D)
        attn_out = self.out_proj(attn_out)

        # Residual connection
        x = query + attn_out

        # FFN with residual
        x = x + self.ffn(self.norm3(x))

        return x


class CrossAttentionFusion(nn.Module):
    """
    Multi-layer Cross-Attention fusion module with spatial downsampling.

    Pipeline:
    1. Downsample both part and whole patch tokens
    2. Apply N layers of Cross-Attention (part queries attend to whole keys/values)
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        target_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.downsample = SpatialDownsample(target_size)
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        part_tokens: torch.Tensor,
        whole_tokens: torch.Tensor,
        grid_size: int,
    ) -> torch.Tensor:
        """
        Args:
            part_tokens: [B, N, D] patch tokens from part encoder
            whole_tokens: [B, N, D] patch tokens from whole encoder
            grid_size: spatial grid side length

        Returns:
            fused_tokens: [B, N', D] attended part tokens
        """
        # Spatial downsampling
        part_ds = self.downsample(part_tokens, grid_size)
        whole_ds = self.downsample(whole_tokens, grid_size)

        # Multi-layer Cross-Attention
        x = part_ds
        for layer in self.layers:
            x = layer(query=x, key_value=whole_ds)

        return x
