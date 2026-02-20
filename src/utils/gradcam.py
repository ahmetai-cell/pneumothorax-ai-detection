"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
Modelin pnömotoraks kararını hangi bölgeye bakarak verdiğini görselleştirir.

Yöntem:
  - Encoder'ın son katmanına forward hook takılır
  - Classification head'in gradyanları backward hook ile yakalanır
  - Global average pooling → ağırlıklar → ısı haritası
  - Isı haritası orijinal görüntüye bindirilir

TÜBİTAK 2209-A | Ahmet Demir
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    PneumothoraxModel için Grad-CAM uygulaması.
    Encoder'ın son feature map'i hedef katman olarak kullanılır.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._features: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Encoder'ın son katmanına forward + backward hook ekler."""
        # segmentation_models_pytorch encoder'ında son blok
        target_layer = self.model.unet.encoder.model.blocks[-1]

        def forward_hook(module, input, output):
            self._features = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Bellek sızıntısını önlemek için hook'ları kaldırır."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Grad-CAM ısı haritası üretir.

        Args:
            image_tensor : (1, 1, H, W) float32, normalize edilmiş
            target_size  : Çıktı boyutu (H, W); None ise orijinal boyut

        Returns:
            heatmap : float32 ndarray (H, W), 0-1 arası normalize
            prob    : Classification olasılığı (0-1)
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        x = image_tensor.to(device)
        x.requires_grad_(False)

        # Forward pass
        seg_pred, cls_pred = self.model(x)
        prob = torch.sigmoid(cls_pred).item()

        # Gradyanları sıfırla ve backward
        self.model.zero_grad()
        cls_pred.backward()

        if self._gradients is None or self._features is None:
            raise RuntimeError("Grad-CAM hook'ları çalışmadı. Modeli kontrol edin.")

        # Global average pooling → kanal ağırlıkları
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Ağırlıklı feature map toplamı
        cam = (weights * self._features).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Boyut ayarla
        if target_size is not None:
            cam = cv2.resize(cam, (target_size[1], target_size[0]))

        return cam, prob


def apply_heatmap(
    gray_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Grad-CAM ısı haritasını gri görüntü üzerine bindirir.

    Args:
        gray_image : uint8 grayscale X-ray (HxW)
        heatmap    : float32 ndarray (HxW), 0-1 arası
        alpha      : Isı haritası yoğunluğu
        colormap   : OpenCV colormap (varsayılan JET)

    Returns:
        result: BGR görüntü, ısı haritası overlay
    """
    # Isı haritasını renklendirmek için 0-255'e ölçekle
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Gri görüntüyü BGR'ye çevir
    if gray_image.ndim == 2:
        bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = gray_image

    # Boyutları eşleştir
    if colored.shape[:2] != bgr.shape[:2]:
        colored = cv2.resize(colored, (bgr.shape[1], bgr.shape[0]))

    return cv2.addWeighted(colored, alpha, bgr, 1.0 - alpha, 0)


def generate_gradcam_result(
    model: torch.nn.Module,
    gray_image: np.ndarray,
    img_size: int = 512,
) -> tuple[np.ndarray, float]:
    """
    Tek görüntü için uçtan uca Grad-CAM üretir.
    FastAPI ve predict.py tarafından çağrılır.

    Args:
        model      : Yüklenmiş PneumothoraxModel
        gray_image : uint8 grayscale X-ray (herhangi boyut)
        img_size   : Modelin beklediği giriş boyutu

    Returns:
        overlay : BGR görüntü (Grad-CAM bindirili)
        prob    : Pnömotoraks olasılığı (0-1)
    """
    # Ön işleme
    resized = cv2.resize(gray_image, (img_size, img_size))
    normalized = resized.astype(np.float32) / 255.0

    # Tensor → (1, 1, H, W)
    tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0)

    cam_obj = GradCAM(model)
    try:
        heatmap, prob = cam_obj.generate(
            tensor, target_size=(gray_image.shape[0], gray_image.shape[1])
        )
    finally:
        cam_obj.remove_hooks()

    overlay = apply_heatmap(gray_image, heatmap, alpha=0.45)
    return overlay, prob
