"""
EfficientNet-B4 Encoder wrapper
TÜBİTAK 2209-A | Ahmet Demir
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


def get_encoder(name="efficientnet-b4", pretrained=True):
    """Return EfficientNet encoder with ImageNet weights."""
    encoder = smp.encoders.get_encoder(
        name,
        in_channels=1,
        depth=5,
        weights="imagenet" if pretrained else None,
    )
    return encoder
