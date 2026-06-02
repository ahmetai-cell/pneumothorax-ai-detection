"""
UNet++ with EfficientNet-B2 encoder — dual head (segmentation + classification).
Deep supervision active during training only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class _AuxSegHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        mid = max(64, in_ch // 2)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            self.head(x), size=target_hw, mode="bilinear", align_corners=False
        )


class PneumothoraxModel(nn.Module):
    """
    UNet++ with EfficientNet-B2 encoder.
    Segmentation head: pixel-wise pneumothorax probability.
    Classification head: image-level binary decision.
    Deep supervision: two auxiliary segmentation heads from encoder features.

    Forward outputs:
        eval  mode: (seg_logits, cls_logit)         shapes: (B,1,H,W), (B,1)
        train mode: (seg_logits, cls_logit, [aux1, aux2])
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b2",
        pretrained: bool = True,
        in_channels: int = 1,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision

        self.unet = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

        enc_chs = self.unet.encoder.out_channels  # e.g. [1, 24, 48, 120, 352]

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(enc_chs[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

        if deep_supervision:
            self.aux_head_1 = _AuxSegHead(enc_chs[-2])
            self.aux_head_2 = _AuxSegHead(enc_chs[-3])

        self._init_classifier()

    def _init_classifier(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        H, W = x.shape[2], x.shape[3]

        features    = self.unet.encoder(x)
        decoder_out = self.unet.decoder(features)
        seg_out     = self.unet.segmentation_head(decoder_out)
        cls_out     = self.classifier(features[-1])

        if self.deep_supervision and self.training:
            aux1 = self.aux_head_1(features[-2], (H, W))
            aux2 = self.aux_head_2(features[-3], (H, W))
            return seg_out, cls_out, [aux1, aux2]

        return seg_out, cls_out

    def count_parameters(self) -> dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def freeze_encoder(self) -> None:
        for p in self.unet.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.unet.encoder.parameters():
            p.requires_grad = True

    def get_param_groups(
        self,
        encoder_lr: float,
        decoder_lr: float,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        encoder_params = list(self.unet.encoder.parameters())
        encoder_ids    = {id(p) for p in encoder_params}
        other_params   = [p for p in self.parameters() if id(p) not in encoder_ids]
        return [
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": other_params,   "lr": decoder_lr, "weight_decay": weight_decay},
        ]
