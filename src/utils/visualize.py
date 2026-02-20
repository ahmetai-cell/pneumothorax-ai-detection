"""
Visualization Utilities
TÜBİTAK 2209-A | Ahmet Demir
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def show_prediction(image, true_mask, pred_mask, threshold=0.5, save_path=None):
    """Display image, ground truth mask, and predicted mask side by side."""
    pred_binary = (torch.sigmoid(pred_mask) > threshold).float()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d1117")

    titles = ["X-Ray Image", "Ground Truth", "Prediction"]
    imgs = [
        image.squeeze().cpu().numpy(),
        true_mask.squeeze().cpu().numpy(),
        pred_binary.squeeze().cpu().numpy(),
    ]

    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, color="white", fontsize=13)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_training_curves(train_losses, val_losses, val_dices, save_path=None):
    """Plot training loss and validation Dice score curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "#00d4ff", label="Train Loss")
    ax1.plot(epochs, val_losses, "#39ff14", label="Val Loss")
    ax1.set_title("Loss Curves", color="white")
    ax1.legend()

    ax2.plot(epochs, val_dices, "#58a6ff", label="Val Dice")
    ax2.set_title("Dice Score", color="white")
    ax2.legend()

    for ax in [ax1, ax2]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
