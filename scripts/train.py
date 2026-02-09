"""
Training entry script for Part Location model.

Usage:
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --config configs/train_config.yaml --resume outputs/checkpoint_epoch_0010.pth
"""

import argparse
import logging
import sys
import os

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part_location.utils.config import load_config
from part_location.models.part_location_model import build_model
from part_location.data.dataset import build_dataloaders
from part_location.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Part Location Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    cfg = load_config(args.config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.warning("Using device: %s", device)

    # Build data loaders
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # Build model
    model = build_model(cfg)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.warning(
        "Model parameters: %d trainable / %d total (%.1f%%)",
        trainable, total, 100.0 * trainable / max(1, total),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )

    # Resume if specified
    if args.resume:
        trainer.resume(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
