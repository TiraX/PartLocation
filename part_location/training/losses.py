"""
Loss functions for Part Location training.

Implements weighted combination of translation L1 loss and scale L1 loss.
"""

import torch
import torch.nn as nn
from typing import Dict


class PartLocationLoss(nn.Module):
    """
    Combined loss for Part Location prediction.

    L_total = w_trans * L1(pred_trans, gt_trans) + w_scale * L1(pred_scale, gt_scale)
    """

    def __init__(self, w_trans: float = 1.0, w_scale: float = 1.0):
        """
        Args:
            w_trans: weight for translation loss
            w_scale: weight for scale loss
        """
        super().__init__()
        self.w_trans = w_trans
        self.w_scale = w_scale
        self.l1 = nn.L1Loss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss.

        Args:
            pred: predicted [B, 4] (tx, ty, tz, scale)
            target: ground truth [B, 4] (tx, ty, tz, scale)

        Returns:
            dict with keys: 'total', 'trans', 'scale'
        """
        pred_trans = pred[:, :3]
        pred_scale = pred[:, 3:]

        gt_trans = target[:, :3]
        gt_scale = target[:, 3:]

        loss_trans = self.l1(pred_trans, gt_trans)
        loss_scale = self.l1(pred_scale, gt_scale)

        loss_total = self.w_trans * loss_trans + self.w_scale * loss_scale

        return {
            "total": loss_total,
            "trans": loss_trans,
            "scale": loss_scale,
        }
