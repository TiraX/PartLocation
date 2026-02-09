"""
Complete Part Location Model.

Assembles: shared DINOv2 encoder -> spatial downsampling -> Cross-Attention fusion
-> global pooling -> MLP prediction head -> 4 outputs (translation[3] + scale[1]).
"""

import torch
import torch.nn as nn

from .encoder import DINOv2Encoder
from .cross_attention import CrossAttentionFusion


class PartLocationModel(nn.Module):
    """
    End-to-end model for predicting part location (translation + scale).

    Architecture:
        1. Shared DINOv2 encoder encodes whole and part images
        2. Cross-Attention fuses part queries with whole context
        3. Global average pooling
        4. MLP head predicts 4 parameters: [tx, ty, tz, scale]
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        embed_dim: int = 768,
        cross_attention_layers: int = 2,
        num_heads: int = 8,
        mlp_hidden_dim: int = 512,
        downsample_size: int = 32,
        image_size: int = 1024,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.image_size = image_size

        # Shared DINOv2 encoder
        self.encoder = DINOv2Encoder(
            backbone_name=backbone_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        encoder_dim = self.encoder.embed_dim

        # Projection layer if encoder dim != embed_dim
        self.proj = None
        if encoder_dim != embed_dim:
            self.proj = nn.Linear(encoder_dim, embed_dim)
            encoder_dim = embed_dim

        # Cross-Attention fusion
        self.cross_attention = CrossAttentionFusion(
            embed_dim=encoder_dim,
            num_layers=cross_attention_layers,
            num_heads=num_heads,
            target_size=downsample_size,
        )

        # MLP prediction head: 4 outputs (tx, ty, tz, scale)
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, 4),
        )

        self.grid_size = self.encoder.get_patch_grid_size(image_size)

    def forward(
        self, whole_image: torch.Tensor, part_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            whole_image: [B, 3, H, W]
            part_image: [B, 3, H, W]

        Returns:
            predictions: [B, 4] (tx, ty, tz, scale)
        """
        # Encode both images with shared encoder
        _, whole_tokens = self.encoder(whole_image)  # [B, N, D]
        _, part_tokens = self.encoder(part_image)    # [B, N, D]

        # Optional projection
        if self.proj is not None:
            whole_tokens = self.proj(whole_tokens)
            part_tokens = self.proj(part_tokens)

        # Cross-Attention fusion
        fused = self.cross_attention(
            part_tokens, whole_tokens, self.grid_size
        )  # [B, N', D]

        # Global average pooling
        pooled = fused.mean(dim=1)  # [B, D]

        # MLP prediction head
        output = self.head(pooled)  # [B, 4]

        return output


def build_model(cfg: dict) -> PartLocationModel:
    """Factory function to build PartLocationModel from config dict."""
    embed_dim = cfg.get("embed_dim", 768)
    num_heads = max(1, embed_dim // 64)

    model = PartLocationModel(
        backbone_name=cfg.get("backbone_name", "dinov2_vitb14"),
        embed_dim=embed_dim,
        cross_attention_layers=cfg.get("cross_attention_layers", 2),
        num_heads=num_heads,
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 512),
        downsample_size=cfg.get("downsample_size", 32),
        image_size=cfg.get("image_size", 1024),
        use_gradient_checkpointing=cfg.get("use_gradient_checkpointing", True),
    )
    return model
