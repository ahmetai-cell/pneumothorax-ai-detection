"""
U-Net++ with Configurable Encoder (EfficientNet-B0 default)
Hybrid: Segmentation head + Classification head

Değişiklik: smp.Unet → smp.UnetPlusPlus (daha derin skip connections,
küçük pnömotoraks odaklarında belirgin doğruluk artışı sağlar)

TÜBİTAK 2209-A | Ahmet Demir
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class PneumothoraxModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b0",
        pretrained: bool = True,
        in_channels: int = 1,
    ):
        super().__init__()

        # ── Segmentation branch: U-Net++ ─────────────────────────────────────
        # UnetPlusPlus, standart U-Net'e göre daha zengin skip connection'lar
        # içerir; küçük lezyonları daha iyi lokalize eder.
        self.unet = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

        # ── Classification branch ────────────────────────────────────────────
        encoder_out_ch = self.unet.encoder.out_channels[-1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_out_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
            seg_out : (B, 1, H, W) — segmentasyon logit'leri
            cls_out : (B, 1)       — sınıflandırma logit'leri
        """
        features = self.unet.encoder(x)
        decoder_out = self.unet.decoder(*features)
        seg_out = self.unet.segmentation_head(decoder_out)
        cls_out = self.classifier(features[-1])
        return seg_out, cls_out
