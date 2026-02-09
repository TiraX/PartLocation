"""
Core training loop for Part Location model.

Implements: AdamW optimizer, cosine annealing scheduler with warmup,
AMP (GradScaler), gradient accumulation, gradient clipping,
TensorBoard logging, checkpoint management.
"""

import logging
import math
import os
import time
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from part_location.models.part_location_model import PartLocationModel
from part_location.training.losses import PartLocationLoss

logger = logging.getLogger(__name__)


class CosineWarmupScheduler:
    """Cosine annealing LR scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        """Update learning rate based on current epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def get_lr(self) -> float:
        """Return current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class Trainer:
    """Handles the complete training pipeline."""

    def __init__(
        self,
        model: PartLocationModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        # Loss function
        self.criterion = PartLocationLoss(
            w_trans=cfg.get("w_trans", 1.0),
            w_scale=cfg.get("w_scale", 1.0),
        )

        # Optimizer (only optimize non-frozen parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )

        # LR scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=cfg.get("warmup_epochs", 5),
            total_epochs=cfg["num_epochs"],
            min_lr=cfg.get("min_lr", 1e-6),
        )

        # AMP
        self.use_amp = cfg.get("use_amp", True)
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient accumulation & clipping
        self.accumulation_steps = cfg.get("accumulation_steps", 1)
        self.grad_clip = cfg.get("grad_clip", 1.0)

        # Checkpoint management
        self.output_dir = cfg["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_every = cfg.get("save_every_epochs", 1)
        self.keep_last_n = cfg.get("keep_last_n", 5)
        self.best_val_loss = float("inf")

        # TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "runs"))

        # State
        self.start_epoch = 0

    def train(self):
        """Run the full training loop."""
        num_epochs = self.cfg["num_epochs"]

        for epoch in range(self.start_epoch, num_epochs):
            self.scheduler.step(epoch)
            current_lr = self.scheduler.get_lr()

            train_losses = self._train_epoch(epoch)
            val_losses = self._validate_epoch(epoch)

            # TensorBoard logging
            self.writer.add_scalar("lr", current_lr, epoch)
            for key, val in train_losses.items():
                self.writer.add_scalar(f"train/{key}", val, epoch)
            for key, val in val_losses.items():
                self.writer.add_scalar(f"val/{key}", val, epoch)

            # Log VRAM peak
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                self.writer.add_scalar("vram_peak_gb", peak_mem, epoch)

            # Checkpoint
            is_best = val_losses["total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses["total"]

            if (epoch + 1) % self.save_every == 0 or is_best:
                self._save_checkpoint(epoch, is_best)

            logger.warning(
                "Epoch %d/%d | LR: %.2e | Train Loss: %.4f | Val Loss: %.4f%s",
                epoch + 1,
                num_epochs,
                current_lr,
                train_losses["total"],
                val_losses["total"],
                " [BEST]" if is_best else "",
            )

        self.writer.close()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        running = {"total": 0.0, "trans": 0.0, "scale": 0.0}
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, (whole_img, part_img, labels) in enumerate(self.train_loader):
            whole_img = whole_img.to(self.device, non_blocking=True)
            part_img = part_img.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                preds = self.model(whole_img, part_img)
                losses = self.criterion(preds, labels)
                loss = losses["total"] / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.grad_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            for k in running:
                running[k] += losses[k].item()
            num_batches += 1

        return {k: v / max(1, num_batches) for k, v in running.items()}

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        running = {"total": 0.0, "trans": 0.0, "scale": 0.0}
        num_batches = 0

        for whole_img, part_img, labels in self.val_loader:
            whole_img = whole_img.to(self.device, non_blocking=True)
            part_img = part_img.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                preds = self.model(whole_img, part_img)
                losses = self.criterion(preds, labels)

            for k in running:
                running[k] += losses[k].item()
            num_batches += 1

        return {k: v / max(1, num_batches) for k, v in running.items()}

    def _save_checkpoint(self, epoch: int, is_best: bool):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
        }

        ckpt_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch:04d}.pth")
        torch.save(state, ckpt_path)

        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(state, best_path)

        # Clean old checkpoints
        self._clean_old_checkpoints()

    def _clean_old_checkpoints(self):
        """Keep only the last N checkpoints (excluding best_model.pth)."""
        ckpts = sorted(
            [
                f
                for f in os.listdir(self.output_dir)
                if f.startswith("checkpoint_epoch_") and f.endswith(".pth")
            ]
        )
        while len(ckpts) > self.keep_last_n:
            old = ckpts.pop(0)
            os.remove(os.path.join(self.output_dir, old))

    def resume(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.warning("Resumed from epoch %d", self.start_epoch)
