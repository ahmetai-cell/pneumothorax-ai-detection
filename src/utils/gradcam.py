"""
Captum Tabanlı Açıklanabilirlik Modülü
  - LayerGradCam    : Hangi bölge tetikledi?
  - IntegratedGradients : Piksel düzeyinde katkı

Eski manuel hook yaklaşımı → Captum ile değiştirildi.
Captum, PyTorch'un resmi XAI kütüphanesidir; daha kararlı ve
genişletilebilir sonuçlar üretir.

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


# ── Hedef katmanı bul ─────────────────────────────────────────────────────────

def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    EfficientNet encoder'ının son blok katmanını döndürür.
    segmentation_models_pytorch encoder hiyerarşisi:
      model.unet.encoder.model.blocks[-1]
    """
    return model.unet.encoder.model.blocks[-1]


# ── Captum Grad-CAM ───────────────────────────────────────────────────────────

def _cls_forward(model, x: torch.Tensor) -> torch.Tensor:
    """Captum'un beklediği tek çıktılı forward wrapper."""
    _, cls_out = model(x)
    return cls_out


def generate_gradcam_captum(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    """
    Captum LayerGradCam ile ısı haritası üretir.

    Args:
        model        : PneumothoraxModel (eval mode)
        image_tensor : (1, 1, H, W) float32
        target_size  : (H, W) çıktı boyutu; None → orijinal boyut

    Returns:
        heatmap : float32 ndarray (H, W), 0-1 normalize
        prob    : Pnömotoraks olasılığı
    """
    if not CAPTUM_AVAILABLE:
        raise ImportError("captum kurulu değil: pip install captum")

    model.eval()
    device = next(model.parameters()).device
    x = image_tensor.to(device).requires_grad_(True)

    target_layer = _get_target_layer(model)
    layer_gc = LayerGradCam(
        forward_func=lambda inp: _cls_forward(model, inp),
        layer=target_layer,
    )

    # Grad-CAM attribution
    attribution = layer_gc.attribute(x, target=None)   # binary → single output
    # (1, C, h, w) → upsample → (1, 1, H, W)
    upsampled = LayerAttribution.interpolate(attribution, x.shape[2:])

    cam = upsampled.squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)  # ReLU
    if cam.max() > 1e-8:
        cam = cam / cam.max()

    if target_size is not None:
        cam = cv2.resize(cam, (target_size[1], target_size[0]))

    with torch.no_grad():
        _, cls_out = model(image_tensor.to(device))
    prob = torch.sigmoid(cls_out).item()

    return cam, prob


# ── Integrated Gradients (piksel düzeyinde katkı) ────────────────────────────

def generate_integrated_gradients(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Integrated Gradients: her pikselin sınıflandırma kararına katkısını gösterir.
    Grad-CAM'den daha ince granülarite sağlar, küçük pnömotoraks için faydalıdır.

    Returns:
        attr_map : float32 ndarray (H, W), normalize edilmiş katkı haritası
    """
    if not CAPTUM_AVAILABLE:
        raise ImportError("captum kurulu değil: pip install captum")

    model.eval()
    device = next(model.parameters()).device
    x = image_tensor.to(device)

    ig = IntegratedGradients(forward_func=lambda inp: _cls_forward(model, inp))
    baseline = torch.zeros_like(x)

    attributions, _ = ig.attribute(
        x, baselines=baseline, n_steps=n_steps, return_convergence_delta=True
    )

    attr_map = attributions.squeeze().detach().cpu().numpy()
    attr_map = np.abs(attr_map)  # büyüklük önemli, işaret değil
    if attr_map.max() > 1e-8:
        attr_map = attr_map / attr_map.max()

    return attr_map


# ── Yedek: Captum yoksa manuel Grad-CAM ──────────────────────────────────────

def _generate_gradcam_manual(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    """Captum kurulu değilse devreye giren yedek implementasyon."""
    model.eval()
    device = next(model.parameters()).device
    x = image_tensor.to(device)

    features_cache: list[torch.Tensor] = []
    grads_cache:    list[torch.Tensor] = []

    target = _get_target_layer(model)

    fh = target.register_forward_hook(lambda m, i, o: features_cache.append(o))
    bh = target.register_full_backward_hook(
        lambda m, gi, go: grads_cache.append(go[0])
    )

    try:
        _, cls_out = model(x)
        prob = torch.sigmoid(cls_out).item()
        model.zero_grad()
        cls_out.backward()
    finally:
        fh.remove()
        bh.remove()

    weights = grads_cache[0].mean(dim=(2, 3), keepdim=True)
    cam = (weights * features_cache[0]).sum(dim=1).squeeze()
    cam = F.relu(cam).detach().cpu().numpy()
    if cam.max() > 1e-8:
        cam = cam / cam.max()

    if target_size:
        cam = cv2.resize(cam, (target_size[1], target_size[0]))

    return cam, prob


# ── Birleşik arayüz ───────────────────────────────────────────────────────────

def generate_gradcam_result(
    model: torch.nn.Module,
    gray_image: np.ndarray,
    img_size: int = 512,
) -> tuple[np.ndarray, float]:
    """
    Tek görüntü için uçtan uca Grad-CAM üretir.
    Captum varsa kullanır, yoksa manuel implementasyona geçer.

    Args:
        model      : Yüklenmiş PneumothoraxModel
        gray_image : uint8 grayscale X-ray (herhangi boyut)
        img_size   : Model giriş boyutu

    Returns:
        overlay : BGR görüntü (ısı haritası bindirili)
        prob    : Pnömotoraks olasılığı (0-1)
    """
    resized    = cv2.resize(gray_image, (img_size, img_size))
    normalized = resized.astype(np.float32) / 255.0
    tensor     = torch.tensor(normalized).unsqueeze(0).unsqueeze(0)

    target_size = (gray_image.shape[0], gray_image.shape[1])

    if CAPTUM_AVAILABLE:
        cam, prob = generate_gradcam_captum(model, tensor, target_size=target_size)
    else:
        cam, prob = _generate_gradcam_manual(model, tensor, target_size=target_size)

    overlay = apply_heatmap(gray_image, cam)
    return overlay, prob


def apply_heatmap(
    gray_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Isı haritasını gri görüntü üzerine bindirir → BGR çıktı."""
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    colored    = cv2.applyColorMap(heatmap_u8, colormap)
    bgr        = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR) if gray_image.ndim == 2 else gray_image
    if colored.shape[:2] != bgr.shape[:2]:
        colored = cv2.resize(colored, (bgr.shape[1], bgr.shape[0]))
    return cv2.addWeighted(colored, alpha, bgr, 1.0 - alpha, 0)
