"""
Trainer: full training loop with AMP, gradient clipping, accumulation,
early stopping, checkpointing, and WandB/TensorBoard logging.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import ProductionLoss
from src.metrics import (
    MetricAccumulator,
    compute_auc,
    compute_sensitivity_specificity,
    dice_score,
    iou_score,
)

logger = logging.getLogger(__name__)


# ── Early Stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max") -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = float("-inf") if mode == "max" else float("inf")
        self.counter   = 0
        self.stop      = False

    def step(self, value: float) -> bool:
        improved = (
            (value > self.best + self.min_delta) if self.mode == "max"
            else (value < self.best - self.min_delta)
        )
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ── Logger wrapper ─────────────────────────────────────────────────────────────

class ExperimentLogger:
    def __init__(self, cfg: dict, fold: int, output_dir: Path) -> None:
        self.fold  = fold
        self.step_ = 0
        self.tb    = None
        self.wb    = None

        if cfg["logging"].get("use_tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = output_dir / cfg["logging"]["log_dir"] / f"fold_{fold}"
                self.tb = SummaryWriter(str(log_dir))
            except ImportError:
                logger.warning("tensorboard not installed; skipping TensorBoard logging")

        if cfg["logging"].get("use_wandb", False):
            try:
                import wandb
                run_name = f"fold_{fold}"
                self.wb  = wandb.init(
                    project=cfg["logging"].get("wandb_project", "ptx"),
                    entity=cfg["logging"].get("wandb_entity") or None,
                    name=run_name,
                    config=cfg,
                    reinit=True,
                )
            except Exception as exc:
                logger.warning(f"WandB init failed: {exc}")

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        s = step if step is not None else self.step_
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, s)
        if self.wb:
            try:
                import wandb
                wandb.log(metrics, step=s)
            except Exception:
                pass
        self.step_ += 1

    def finish(self) -> None:
        if self.tb:
            self.tb.close()
        if self.wb:
            try:
                self.wb.finish()
            except Exception:
                pass


# ── Trainer ────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Encapsulates one fold of training.

    Usage:
        trainer = Trainer(model, cfg, device, output_dir, fold=1)
        best_metrics = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model:       nn.Module,
        cfg:         dict,
        device:      torch.device,
        output_dir:  Path,
        fold:        int = 1,
    ) -> None:
        self.model      = model.to(device)
        self.cfg        = cfg
        self.device     = device
        self.output_dir = output_dir
        self.fold       = fold

        t_cfg = cfg["training"]

        self.loss_fn = ProductionLoss(
            tversky_alpha  = t_cfg["loss"]["tversky_alpha"],
            tversky_beta   = t_cfg["loss"]["tversky_beta"],
            focal_alpha    = t_cfg["loss"].get("focal_alpha", 0.25),
            focal_gamma    = t_cfg["loss"].get("focal_gamma", 2.0),
            seg_tversky_w  = t_cfg["loss"]["seg_tversky_w"],
            seg_focal_w    = t_cfg["loss"]["seg_focal_w"],
            cls_w          = t_cfg["loss"]["cls_w"],
            aux_w          = t_cfg["loss"]["aux_w"],
        )

        opt_cfg = t_cfg["optimizer"]
        self.optimizer = torch.optim.AdamW(
            self.model.get_param_groups(
                encoder_lr   = opt_cfg["encoder_lr"],
                decoder_lr   = opt_cfg["decoder_lr"],
                weight_decay = opt_cfg["weight_decay"],
            )
        )

        sch_cfg = t_cfg["scheduler"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0    = sch_cfg["T_0"],
            T_mult = sch_cfg["T_mult"],
            eta_min = sch_cfg["eta_min"],
        )

        # AMP only on CUDA; MPS/CPU use full precision.
        self.use_amp     = t_cfg.get("amp", True) and device.type == "cuda"
        self.scaler      = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_clip   = t_cfg.get("grad_clip", 1.0)
        self.accum_steps = t_cfg.get("grad_accum_steps", 1)
        self.epochs      = t_cfg.get("epochs", 80)

        es_cfg           = t_cfg.get("early_stopping", {})
        self.early_stop  = EarlyStopping(
            patience  = es_cfg.get("patience", 15),
            min_delta = es_cfg.get("min_delta", 1e-4),
            mode      = "max",
        )

        self.exp_logger = ExperimentLogger(cfg, fold, output_dir)
        self.ckpt_dir   = output_dir / cfg["checkpoint"]["dir"]
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_dice  = 0.0
        self.best_epoch = 0

    # ── Train epoch ───────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        acc = MetricAccumulator()

        self.optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"  [E{epoch}] Train", leave=False, unit="batch")

        for batch_i, (images, masks, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)

                if len(outputs) == 3:
                    seg_pred, cls_pred, aux_preds = outputs
                else:
                    seg_pred, cls_pred = outputs
                    aux_preds = None

                total_loss, components = self.loss_fn(
                    seg_pred, masks, cls_pred, labels, aux_preds
                )
                total_loss = total_loss / self.accum_steps

            self.scaler.scale(total_loss).backward()

            if (batch_i + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            acc.update({
                "loss":     total_loss.item() * self.accum_steps,
                "seg_loss": components["seg_loss"],
                "cls_loss": components["cls_loss"],
            }, n=len(images))

            pbar.set_postfix(loss=f"{total_loss.item() * self.accum_steps:.4f}")

        # Handle remaining accumulation steps
        if len(loader) % self.accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        self.scheduler.step(epoch)
        return acc.mean()

    # ── Validation epoch ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        acc = MetricAccumulator()

        all_cls_probs:  list[float] = []
        all_cls_labels: list[int]   = []
        all_dices:      list[float] = []
        all_ious:       list[float] = []

        pbar = tqdm(loader, desc="         Val  ", leave=False, unit="batch")

        for images, masks, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                seg_pred, cls_pred = self.model(images)
                total_loss, components = self.loss_fn(
                    seg_pred, masks, cls_pred, labels, None
                )

            acc.update({
                "loss":     total_loss.item(),
                "seg_loss": components["seg_loss"],
                "cls_loss": components["cls_loss"],
            }, n=len(images))

            seg_probs  = torch.sigmoid(seg_pred).cpu().numpy()
            cls_probs_ = torch.sigmoid(cls_pred).squeeze(-1).cpu().numpy()
            masks_np   = masks.cpu().numpy()
            labels_np  = labels.cpu().numpy().astype(int)

            for i, lbl in enumerate(labels_np):
                cls_p = float(cls_probs_[i]) if cls_probs_.ndim > 0 else float(cls_probs_)
                all_cls_probs.append(cls_p)
                all_cls_labels.append(lbl)

                if lbl == 1:
                    d = dice_score(seg_probs[i, 0], masks_np[i, 0])
                    u = iou_score(seg_probs[i, 0], masks_np[i, 0])
                    all_dices.append(d)
                    all_ious.append(u)

        probs_arr  = np.array(all_cls_probs)
        labels_arr = np.array(all_cls_labels)

        auc = compute_auc(probs_arr, labels_arr)
        sensitivity, specificity = compute_sensitivity_specificity(
            probs_arr, labels_arr, threshold=0.5
        )

        result = acc.mean()
        result.update({
            "dice":        float(np.mean(all_dices))  if all_dices else 0.0,
            "iou":         float(np.mean(all_ious))   if all_ious  else 0.0,
            "auc":         auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        })
        return result

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, metrics: dict, tag: str) -> Path:
        path = self.ckpt_dir / f"fold_{self.fold}_{tag}.pth"
        torch.save({
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "sched_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "metrics":     metrics,
            "fold":        self.fold,
        }, str(path))
        return path

    def _load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(str(path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.scheduler.load_state_dict(ckpt["sched_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        logger.info(f"Resumed from epoch {ckpt['epoch']} (fold {ckpt['fold']})")
        return int(ckpt["epoch"])

    # ── Main fit loop ──────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        resume:       bool = False,
    ) -> dict[str, float]:

        start_epoch = 0

        if resume:
            last_ckpt = self.ckpt_dir / f"fold_{self.fold}_last.pth"
            if last_ckpt.exists():
                start_epoch = self._load_checkpoint(last_ckpt) + 1
                logger.info(f"Resumed fold {self.fold} from epoch {start_epoch}")
            else:
                logger.warning(f"--resume given but no checkpoint found: {last_ckpt}")

        logger.info(
            f"\n{'='*60}\n  Fold {self.fold} — training epochs {start_epoch+1}..{self.epochs}\n"
            f"  Device: {self.device}  AMP: {self.use_amp}\n{'='*60}"
        )

        for epoch in range(start_epoch, self.epochs):
            t0 = time.time()

            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics   = self._val_epoch(val_loader)

            elapsed = time.time() - t0
            lr_enc  = self.optimizer.param_groups[0]["lr"]
            lr_dec  = self.optimizer.param_groups[1]["lr"]

            logger.info(
                f"  E{epoch+1:03d}/{self.epochs} | "
                f"loss={train_metrics['loss']:.4f} | "
                f"val_dice={val_metrics['dice']:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | "
                f"val_sens={val_metrics['sensitivity']:.4f} | "
                f"val_spec={val_metrics['specificity']:.4f} | "
                f"lr_enc={lr_enc:.2e} | "
                f"t={elapsed:.0f}s"
            )

            log_data = {
                f"fold_{self.fold}/train_loss":    train_metrics["loss"],
                f"fold_{self.fold}/val_loss":      val_metrics["loss"],
                f"fold_{self.fold}/val_dice":      val_metrics["dice"],
                f"fold_{self.fold}/val_iou":       val_metrics["iou"],
                f"fold_{self.fold}/val_auc":       val_metrics["auc"],
                f"fold_{self.fold}/val_sens":      val_metrics["sensitivity"],
                f"fold_{self.fold}/val_spec":      val_metrics["specificity"],
                f"fold_{self.fold}/lr_encoder":    lr_enc,
                f"fold_{self.fold}/lr_decoder":    lr_dec,
            }
            self.exp_logger.log(log_data, step=epoch)

            # Best checkpoint
            if val_metrics["dice"] > self.best_dice:
                self.best_dice  = val_metrics["dice"]
                self.best_epoch = epoch + 1
                self._save_checkpoint(epoch, val_metrics, "best")
                logger.info(f"  → New best Dice: {self.best_dice:.4f} (epoch {self.best_epoch})")

            # Last checkpoint (for resume)
            self._save_checkpoint(epoch, val_metrics, "last")

            # Early stopping
            if self.early_stop.step(val_metrics["dice"]):
                logger.info(
                    f"  Early stopping triggered at epoch {epoch+1} "
                    f"(best={self.best_dice:.4f} @ {self.best_epoch})"
                )
                break

        self.exp_logger.finish()

        best_ckpt = self.ckpt_dir / f"fold_{self.fold}_best.pth"
        best_ckpt_data = torch.load(str(best_ckpt), map_location="cpu", weights_only=True)
        return best_ckpt_data["metrics"]
