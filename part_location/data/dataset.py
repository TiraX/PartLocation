"""
Dataset module for Part Location training.

Scans data/images/ sub-directories, matches whole images, part images and
JSON annotations, and builds (whole, part, label) triplets.
"""

import json
import logging
import os
import math
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)


class PartLocationDataset(Dataset):
    """
    PyTorch Dataset that yields (whole_image, part_image, label) triplets.

    Label is a 4-dim tensor: [tx, ty, tz, scale].
    """

    def __init__(
        self,
        samples: List[Tuple[str, str, List[float]]],
        image_size: int = 1024,
        augment: bool = False,
    ):
        """
        Args:
            samples: list of (whole_image_path, part_image_path, [tx, ty, tz, scale])
            image_size: target image size for resize
            augment: whether to apply training-time data augmentation
        """
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

        # Build transforms
        self._transform = self._build_transform(augment)

    def _build_transform(self, augment: bool) -> transforms.Compose:
        """Build image transform pipeline."""
        t = []
        t.append(transforms.Resize((self.image_size, self.image_size)))

        if augment:
            t.append(
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                )
            )

        t.append(transforms.ToTensor())
        # ImageNet normalization
        t.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ))
        return transforms.Compose(t)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        whole_path, part_path, label = self.samples[idx]

        whole_img = Image.open(whole_path).convert("RGB")
        part_img = Image.open(part_path).convert("RGB")

        whole_img = self._transform(whole_img)
        part_img = self._transform(part_img)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return whole_img, part_img, label_tensor


def scan_dataset(data_dir: str) -> Dict[str, List[Tuple[str, str, List[float]]]]:
    """
    Scan data directory and build samples grouped by model name.

    Returns:
        dict mapping model_name -> list of (whole_path, part_path, [tx, ty, tz, scale])
    """
    model_samples: Dict[str, List[Tuple[str, str, List[float]]]] = {}

    if not os.path.isdir(data_dir):
        logger.warning("Data directory not found: %s", data_dir)
        return model_samples

    for model_name in sorted(os.listdir(data_dir)):
        model_dir = os.path.join(data_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        # Find whole image
        whole_image_path = os.path.join(model_dir, f"{model_name}-whole.png")
        if not os.path.isfile(whole_image_path):
            logger.warning(
                "Whole image not found for model '%s', skipping entire directory.",
                model_name,
            )
            continue

        samples = []
        # Find all JSON annotation files
        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(".json"):
                continue

            json_path = os.path.join(model_dir, fname)
            # Derive the corresponding part image path
            part_base = fname[:-5]  # Remove .json
            part_image_path = os.path.join(model_dir, part_base + ".png")

            if not os.path.isfile(part_image_path):
                logger.warning(
                    "Part image not found for annotation '%s', skipping.", json_path
                )
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    anno = json.load(f)

                translation = anno["translation"]  # [tx, ty, tz]
                scale_arr = anno["scale"]  # [s, s, s]
                scale = scale_arr[0]  # Take first element (uniform scale)

                label = [translation[0], translation[1], translation[2], scale]
                samples.append((whole_image_path, part_image_path, label))

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.warning(
                    "Failed to parse annotation '%s': %s. Skipping.", json_path, e
                )
                continue

        if samples:
            model_samples[model_name] = samples
        else:
            logger.warning("No valid samples found for model '%s'.", model_name)

    return model_samples


def split_by_model(
    model_samples: Dict[str, List[Tuple[str, str, List[float]]]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[
    List[Tuple[str, str, List[float]]],
    List[Tuple[str, str, List[float]]],
    List[Tuple[str, str, List[float]]],
]:
    """
    Split samples by model to prevent data leakage.
    All parts of the same model go into the same split.

    Returns:
        (train_samples, val_samples, test_samples)
    """
    import random

    model_names = sorted(model_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(model_names)

    n = len(model_names)
    n_train = max(1, math.floor(n * train_ratio))
    n_val = max(0, math.floor(n * val_ratio))
    # Remaining goes to test
    n_test = n - n_train - n_val

    train_models = model_names[:n_train]
    val_models = model_names[n_train : n_train + n_val]
    test_models = model_names[n_train + n_val :]

    train_samples = []
    val_samples = []
    test_samples = []

    for m in train_models:
        train_samples.extend(model_samples[m])
    for m in val_models:
        val_samples.extend(model_samples[m])
    for m in test_models:
        test_samples.extend(model_samples[m])

    logger.warning(
        "Dataset split: %d models -> train=%d(%d samples), val=%d(%d samples), test=%d(%d samples)",
        n,
        len(train_models), len(train_samples),
        len(val_models), len(val_samples),
        len(test_models), len(test_samples),
    )

    return train_samples, val_samples, test_samples


def build_dataloaders(
    cfg: dict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Build train, val, and test DataLoaders from config.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    model_samples = scan_dataset(cfg["data_dir"])

    if not model_samples:
        raise RuntimeError(f"No valid samples found in {cfg['data_dir']}")

    train_samples, val_samples, test_samples = split_by_model(
        model_samples,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        seed=cfg.get("split_seed", 42),
    )

    train_ds = PartLocationDataset(
        train_samples, image_size=cfg["image_size"], augment=True
    )
    val_ds = PartLocationDataset(
        val_samples, image_size=cfg["image_size"], augment=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=cfg.get("pin_memory", True),
        drop_last=False,
    )

    test_loader = None
    if test_samples:
        test_ds = PartLocationDataset(
            test_samples, image_size=cfg["image_size"], augment=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=cfg.get("pin_memory", True),
            drop_last=False,
        )

    return train_loader, val_loader, test_loader
