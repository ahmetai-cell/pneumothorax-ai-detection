"""
U-Net++ with Configurable Encoder (EfficientNet-B0 default)
Hybrid: Segmentation head + Classification head

Değişiklik: smp.Unet → smp.UnetPlusPlus (daha derin skip connections,
küçük pnömotoraks odaklarında belirgin doğruluk artışı sağlar)

TÜBİTAK 2209-A | Ahmet Demir, Erkan Koçulu
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class _AuxSegHead(nn.Module):
    """
    Encoder feature map'inden auxiliary segmentasyon çıktısı.
    Deep Supervision: farklı derinliklerdeki encoder katmanlarına
    doğrudan gradyan akışı sağlar — ince plevral hat öğrenimini destekler.
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        return nn.functional.interpolate(
            self.head(x), size=target_hw, mode="bilinear", align_corners=False
        )


class PneumothoraxModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b0",
        pretrained: bool = True,
        in_channels: int = 1,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # ── Segmentation branch: U-Net++ ─────────────────────────────────────
        self.unet = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

        # ── Classification branch ────────────────────────────────────────────
        enc_chs = self.unet.encoder.out_channels  # örn. [3, 32, 48, 136, 384]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(enc_chs[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        # ── Deep Supervision: encoder'ın -2. ve -3. katmanlarından aux head ──
        # Gradyan farklı derinliklerden akarak plevral hat gibi ince
        # yapıların kaybolmadan öğrenilmesini sağlar.
        if deep_supervision:
            self.aux_head_1 = _AuxSegHead(enc_chs[-2])  # derin encoder katmanı
            self.aux_head_2 = _AuxSegHead(enc_chs[-3])  # orta encoder katmanı

    def forward(self, x: torch.Tensor):
        """
        Eğitim: (seg_out, cls_out, [aux1, aux2])
        Inference: (seg_out, cls_out)
        """
        H, W        = x.shape[2], x.shape[3]
        features    = self.unet.encoder(x)
        decoder_out = self.unet.decoder(features)
        seg_out     = self.unet.segmentation_head(decoder_out)
        cls_out     = self.classifier(features[-1])

        if self.deep_supervision and self.training:
            # Auxiliary head'ler SADECE eğitimde aktif.
            # model.eval() çağrıldığı anda (inference) self.training=False olur
            # → bu blok çalışmaz, GPU belleği harcanmaz, latency sıfır.
            aux1 = self.aux_head_1(features[-2], (H, W))
            aux2 = self.aux_head_2(features[-3], (H, W))
            return seg_out, cls_out, [aux1, aux2]

        # Inference: sadece ana çıkışlar döner
        return seg_out, cls_out
