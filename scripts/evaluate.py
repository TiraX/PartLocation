"""
Evaluation entry script for Part Location model.

Runs model on test set and outputs a detailed evaluation report.

Usage:
    python scripts/evaluate.py --config configs/train_config.yaml \
        --checkpoint outputs/best_model.pth \
        --output eval_report.json
"""

import argparse
import json
import logging
import sys
import os

import numpy as np
import torch
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part_location.utils.config import load_config
from part_location.models.part_location_model import build_model
from part_location.data.dataset import build_dataloaders
from part_location.evaluation.metrics import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate Part Location Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON report file path"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config and checkpoint
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt.get("config", cfg)
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Build data loaders (we need test set)
    _, _, test_loader = build_dataloaders(cfg)

    if test_loader is None:
        logging.error("No test samples available. Check data_dir and split ratios.")
        sys.exit(1)

    # Collect predictions
    all_preds = []
    all_targets = []
    use_amp = cfg.get("use_amp", True)

    with torch.no_grad():
        for whole_img, part_img, labels in test_loader:
            whole_img = whole_img.to(device, non_blocking=True)
            part_img = part_img.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                preds = model(whole_img, part_img)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Evaluate
    report = evaluate_predictions(all_preds, all_targets)

    # Print summary
    summary = report["summary"]
    logging.warning("=== Evaluation Report ===")
    logging.warning(
        "Mean Translation Error (L2): %.6f", summary["mean_translation_error"]
    )
    logging.warning(
        "Mean Scale Absolute Error: %.6f", summary["mean_scale_absolute_error"]
    )
    logging.warning(
        "Mean Scale Relative Error: %.2f%%", summary["mean_scale_relative_pct_error"]
    )
    logging.warning("Total test samples: %d", len(all_preds))

    # Save report
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logging.warning("Full report saved to %s", args.output)
    else:
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
