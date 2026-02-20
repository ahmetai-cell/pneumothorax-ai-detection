"""
U-Net with EfficientNet-B4 Encoder
TÜBİTAK 2209-A | Pneumothorax Detection
Author: Ahmet Demir
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class PneumothoraxModel(nn.Module):
    def __init__(self, encoder_name="efficientnet-b4", pretrained=True):
        super().__init__()

        # Segmentation branch (U-Net)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=1,
            classes=1,
            activation=None,
        )

        # Classification branch
        encoder_out_channels = self.unet.encoder.out_channels[-1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_out_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.unet.encoder(x)
        seg_mask = self.unet.decoder(*features)
        seg_out = self.unet.segmentation_head(seg_mask)
        cls_out = self.classifier(features[-1])
        return seg_out, cls_out
