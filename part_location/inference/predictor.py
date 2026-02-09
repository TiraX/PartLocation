"""
Inference predictor for Part Location model.

Handles model loading and single/batch prediction.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import torch
from torchvision import transforms
from PIL import Image

from part_location.models.part_location_model import PartLocationModel, build_model

logger = logging.getLogger(__name__)


class Predictor:
    """Loads a trained Part Location model and performs inference."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            checkpoint_path: path to .pth checkpoint file
            device: torch device (auto-detected if None)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = ckpt["config"]

        # Build model from saved config
        self.model = build_model(cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        self.image_size = cfg.get("image_size", 1024)

        # Standard inference transform
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def predict_single(
        self, whole_image_path: str, part_image_path: str
    ) -> Dict[str, object]:
        """
        Predict translation and scale for a single (whole, part) pair.

        Returns:
            dict with 'translation' [3] and 'scale' (float)
        """
        whole_img = Image.open(whole_image_path).convert("RGB")
        part_img = Image.open(part_image_path).convert("RGB")

        whole_tensor = self.transform(whole_img).unsqueeze(0).to(self.device)
        part_tensor = self.transform(part_img).unsqueeze(0).to(self.device)

        output = self.model(whole_tensor, part_tensor)  # [1, 4]
        output = output.cpu().squeeze(0).tolist()

        return {
            "translation": output[:3],
            "scale": output[3],
        }

    @torch.no_grad()
    def predict_batch(self, data_dir: str) -> List[Dict]:
        """
        Predict for all models in a data directory.

        Args:
            data_dir: path to data/images/ directory

        Returns:
            list of result dicts
        """
        results = []

        for model_name in sorted(os.listdir(data_dir)):
            model_dir = os.path.join(data_dir, model_name)
            if not os.path.isdir(model_dir):
                continue

            whole_path = os.path.join(model_dir, f"{model_name}-whole.png")
            if not os.path.isfile(whole_path):
                logger.warning("No whole image for '%s', skipping.", model_name)
                continue

            for fname in sorted(os.listdir(model_dir)):
                if not fname.endswith(".png") or fname.endswith("-whole.png"):
                    continue

                part_path = os.path.join(model_dir, fname)
                part_name = fname[len(model_name) + 1 : -4]  # Remove prefix and .png

                try:
                    result = self.predict_single(whole_path, part_path)
                    result["model_name"] = model_name
                    result["part_name"] = part_name
                    results.append(result)
                except Exception as e:
                    logger.warning(
                        "Failed to predict for %s/%s: %s", model_name, fname, e
                    )

        return results
