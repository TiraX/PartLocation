"""
Configuration loading and validation utilities.
"""

import os
import yaml
import logging

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULTS = {
    "data_dir": "./data/images",
    "output_dir": "./outputs",
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "split_seed": 42,
    "backbone_name": "dinov2_vitb14",
    "cross_attention_layers": 2,
    "mlp_hidden_dim": 512,
    "downsample_size": 32,
    "image_size": 1024,
    "batch_size": 2,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 0.05,
    "grad_clip": 1.0,
    "accumulation_steps": 1,
    "use_amp": True,
    "use_gradient_checkpointing": True,
    "num_workers": 4,
    "pin_memory": True,
    "w_trans": 1.0,
    "w_scale": 1.0,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    "save_every_epochs": 1,
    "keep_last_n": 5,
}

# Backbone name to embed_dim mapping
BACKBONE_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


def load_config(config_path: str) -> dict:
    """Load a YAML config file, fill missing keys with defaults, and validate."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Fill defaults for missing keys
    for key, default_val in DEFAULTS.items():
        if key not in cfg:
            cfg[key] = default_val
            logger.warning("Config key '%s' not found, using default: %s", key, default_val)

    # Derive embed_dim from backbone_name
    backbone = cfg["backbone_name"]
    if backbone not in BACKBONE_EMBED_DIMS:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. "
            f"Choose from: {list(BACKBONE_EMBED_DIMS.keys())}"
        )
    cfg["embed_dim"] = BACKBONE_EMBED_DIMS[backbone]

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    """Validate configuration values."""
    # Check split ratios
    total = cfg["train_ratio"] + cfg["val_ratio"] + cfg["test_ratio"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}"
        )

    # Check positive values
    for key in ["batch_size", "num_epochs", "image_size", "downsample_size"]:
        if cfg[key] <= 0:
            raise ValueError(f"'{key}' must be positive, got {cfg[key]}")

    # Check learning rate
    if cfg["learning_rate"] <= 0:
        raise ValueError(f"'learning_rate' must be positive, got {cfg['learning_rate']}")

    # Check accumulation steps
    if cfg["accumulation_steps"] < 1:
        raise ValueError(
            f"'accumulation_steps' must be >= 1, got {cfg['accumulation_steps']}"
        )

    # Check loss weights are non-negative
    for key in ["w_trans", "w_scale"]:
        if cfg[key] < 0:
            raise ValueError(f"'{key}' must be non-negative, got {cfg[key]}")
