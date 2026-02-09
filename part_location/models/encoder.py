"""
DINOv2 ViT encoder wrapper.

Loads DINOv2 pretrained weights via torch.hub and provides access to
CLS token and patch tokens. Supports Gradient Checkpointing.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

# Mapping from config name to torch.hub model name
_DINOV2_HUB_MODELS = {
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_vitb14": "dinov2_vitb14",
    "dinov2_vitl14": "dinov2_vitl14",
}


class DINOv2Encoder(nn.Module):
    """
    Wrapper around a DINOv2 ViT model.

    Returns CLS token and patch tokens from the final layer.
    Optionally enables Gradient Checkpointing on transformer blocks.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        if backbone_name not in _DINOV2_HUB_MODELS:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Choose from: {list(_DINOV2_HUB_MODELS.keys())}"
            )

        hub_name = _DINOV2_HUB_MODELS[backbone_name]
        self.model = torch.hub.load("facebookresearch/dinov2", hub_name)

        # Freeze the pretrained parameters by default
        for param in self.model.parameters():
            param.requires_grad = False

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size

        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for transformer blocks."""
        if hasattr(self.model, "blocks"):
            for block in self.model.blocks:
                block._original_forward = block.forward

                def make_ckpt_forward(blk):
                    def ckpt_forward(x):
                        return checkpoint(blk._original_forward, x, use_reentrant=False)
                    return ckpt_forward

                block.forward = make_ckpt_forward(block)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images and return CLS token + patch tokens.

        Args:
            x: input images [B, 3, H, W]

        Returns:
            cls_token: [B, D]
            patch_tokens: [B, N, D] where N = (H/patch_size) * (W/patch_size)
        """
        # DINOv2 forward_features returns a dict with keys:
        # 'x_norm_clstoken', 'x_norm_patchtokens', etc.
        features = self.model.forward_features(x)

        cls_token = features["x_norm_clstoken"]      # [B, D]
        patch_tokens = features["x_norm_patchtokens"]  # [B, N, D]

        return cls_token, patch_tokens

    def get_num_patches(self, image_size: int) -> int:
        """Calculate number of patches for a given image size."""
        return (image_size // self.patch_size) ** 2

    def get_patch_grid_size(self, image_size: int) -> int:
        """Calculate the spatial grid size (H_patches = W_patches)."""
        return image_size // self.patch_size
