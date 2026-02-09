"""
Evaluation metrics for Part Location predictions.

Computes translation error (L2 distance) and scale error (absolute + relative).
"""

import torch
import numpy as np
from typing import Dict, List, Optional


def compute_translation_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Compute per-sample L2 distance between predicted and ground-truth translations.

    Args:
        pred: [N, 3] predicted translations
        gt: [N, 3] ground truth translations

    Returns:
        [N] array of L2 distances
    """
    return np.linalg.norm(pred - gt, axis=1)


def compute_scale_error(pred: np.ndarray, gt: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute per-sample scale errors (absolute and relative percentage).

    Args:
        pred: [N] or [N, 1] predicted scales
        gt: [N] or [N, 1] ground truth scales

    Returns:
        dict with 'absolute' [N] and 'relative_pct' [N] arrays
    """
    pred = pred.flatten()
    gt = gt.flatten()

    absolute = np.abs(pred - gt)
    # Avoid division by zero for relative error
    safe_gt = np.where(np.abs(gt) < 1e-8, 1e-8, gt)
    relative_pct = (absolute / np.abs(safe_gt)) * 100.0

    return {"absolute": absolute, "relative_pct": relative_pct}


def evaluate_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    sample_names: Optional[List[str]] = None,
) -> Dict:
    """
    Run full evaluation on a batch of predictions.

    Args:
        preds: [N, 4] predictions (tx, ty, tz, scale)
        targets: [N, 4] ground truth (tx, ty, tz, scale)
        sample_names: optional list of sample identifiers

    Returns:
        dict containing:
            - 'summary': aggregated metrics
            - 'per_sample': list of per-sample metric dicts
    """
    pred_trans = preds[:, :3]
    gt_trans = targets[:, :3]
    pred_scale = preds[:, 3]
    gt_scale = targets[:, 3]

    trans_errors = compute_translation_error(pred_trans, gt_trans)
    scale_errors = compute_scale_error(pred_scale, gt_scale)

    summary = {
        "mean_translation_error": float(np.mean(trans_errors)),
        "std_translation_error": float(np.std(trans_errors)),
        "median_translation_error": float(np.median(trans_errors)),
        "max_translation_error": float(np.max(trans_errors)),
        "mean_scale_absolute_error": float(np.mean(scale_errors["absolute"])),
        "std_scale_absolute_error": float(np.std(scale_errors["absolute"])),
        "mean_scale_relative_pct_error": float(np.mean(scale_errors["relative_pct"])),
        "std_scale_relative_pct_error": float(np.std(scale_errors["relative_pct"])),
    }

    per_sample = []
    for i in range(len(preds)):
        entry = {
            "translation_error": float(trans_errors[i]),
            "scale_absolute_error": float(scale_errors["absolute"][i]),
            "scale_relative_pct_error": float(scale_errors["relative_pct"][i]),
            "pred_translation": preds[i, :3].tolist(),
            "gt_translation": targets[i, :3].tolist(),
            "pred_scale": float(preds[i, 3]),
            "gt_scale": float(targets[i, 3]),
        }
        if sample_names is not None:
            entry["name"] = sample_names[i]
        per_sample.append(entry)

    return {"summary": summary, "per_sample": per_sample}
