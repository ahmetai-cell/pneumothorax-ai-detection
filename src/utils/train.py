"""
Training Loop
TÜBİTAK 2209-A | Ahmet Demir
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.unet import PneumothoraxModel
from src.model.losses import CombinedLoss
from src.utils.metrics import dice_score, iou_score


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice = 0, 0

    for images, masks, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        seg_pred, cls_pred = model(images)
        loss = criterion(seg_pred, masks, cls_pred, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(seg_pred, masks).item()

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0

    for images, masks, labels in tqdm(loader, desc="Validation"):
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        seg_pred, cls_pred = model(images)
        loss = criterion(seg_pred, masks, cls_pred, labels)

        total_loss += loss.item()
        total_dice += dice_score(seg_pred, masks).item()
        total_iou  += iou_score(seg_pred, masks).item()

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PneumothoraxModel().to(device)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_dice = 0
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        train_loss, train_dice = train_epoch(model, config["train_loader"], optimizer, criterion, device)
        val_loss, val_dice, val_iou = val_epoch(model, config["val_loader"], criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
        print(f"Val   Loss: {val_loss:.4f}  | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "results/checkpoints/best_model.pth")
            print(f"✓ Best model saved (Dice: {best_dice:.4f})")

    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")
