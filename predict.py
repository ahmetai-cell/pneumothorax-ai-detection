"""
Single Image Prediction — Grad-CAM destekli
Kullanım:
    python predict.py --image path/to/xray.png
    python predict.py --image path/to/xray.png --model results/checkpoints/best_model.pth

TÜBİTAK 2209-A | Ahmet Demir
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.model.unet import PneumothoraxModel
from src.utils.gradcam import generate_gradcam_result


def predict(image_path: str, model_path: str = "results/checkpoints/best_model.pth", threshold: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model yükle
    model = PneumothoraxModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # Görüntüyü yükle
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")

    # Segmentasyon + sınıflandırma
    resized = cv2.resize(gray, (512, 512))
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_pred, cls_pred = model(tensor)

    prob = torch.sigmoid(cls_pred).item()
    seg_mask = (torch.sigmoid(seg_pred) > threshold).float().squeeze().cpu().numpy()

    # Grad-CAM
    gradcam_bgr, _ = generate_gradcam_result(model, gray, img_size=512)
    gradcam_rgb = cv2.cvtColor(gradcam_bgr, cv2.COLOR_BGR2RGB)

    # Sonuç
    label = "PNÖMOTORAKS TESPİT EDİLDİ" if prob >= threshold else "Normal"
    print(f"Tanı       : {label}")
    print(f"Olasılık   : {prob:.4f}  ({prob:.1%})")

    # Görselleştir (3 panel)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0d1117")

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Giriş X-Ray", color="white")

    axes[1].imshow(seg_mask, cmap="Reds")
    axes[1].set_title(f"Segmentasyon Maskesi\nP={prob:.3f}", color="white")

    axes[2].imshow(gradcam_rgb)
    axes[2].set_title("Grad-CAM Isı Haritası", color="white")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"{label} — Olasılık: {prob:.1%}", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/figures/prediction.png", dpi=150, bbox_inches="tight")
    print("Sonuç kaydedildi: results/figures/prediction.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Akciğer grafisi pnömotoraks analizi")
    parser.add_argument("--image",     required=True, help="Görüntü dosya yolu (PNG/JPEG/DICOM)")
    parser.add_argument("--model",     default="results/checkpoints/best_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    predict(args.image, args.model, args.threshold)
