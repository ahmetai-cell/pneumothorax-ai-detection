"""
Green Annotation Mask Extractor
Tıp Fakültesi'nden gelen yeşil boyalı akciğer grafilerinden:
  1. Binary pnömotoraks maskesi çıkarır
  2. Orijinal gri görüntüyü kurtarır

Yeşil boya sadece G kanalını etkiler; R ve B kanalları
orijinal grayscale değerlerine yakın kalır. Bu özellik
hem opak hem yarı saydam overlay durumunda çalışır.

TÜBİTAK 2209-A | Ahmet Demir
"""

import cv2
import numpy as np
import pydicom
from pathlib import Path


# HSV renk uzayında yeşil aralığı
# H: 35-85  → sarı-yeşil-cyan arası
# S: 40-255 → yeterince doygun (beyaz/gri alanları dışlar)
# V: 40-255 → yeterince parlak
GREEN_HSV_LOWER = np.array([35, 40, 40])
GREEN_HSV_UPPER = np.array([85, 255, 255])


def load_image(path: str) -> np.ndarray:
    """
    DICOM veya PNG/JPG görüntüyü BGR formatında yükler.
    DICOM için piksel değerleri 0-255 aralığına normalize edilir.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".dcm", ".dicom"):
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        # Grayscale DICOM → BGR (annotasyon renkli PNG olarak üstüne gelir)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr
    else:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {path}")
        return img


def extract_green_mask(bgr_image: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    Yeşil boyalı bölgelerden binary maske çıkarır.

    Args:
        bgr_image : OpenCV BGR görüntü (uint8)
        min_area  : Gürültü temizleme için minimum bileşen alanı (piksel)

    Returns:
        mask: uint8 ndarray, 255 = pnömotoraks, 0 = normal doku
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)

    # Morfolojik temizlik:
    # CLOSE → küçük delikleri kapat
    # OPEN  → ince gürültü noktalarını sil
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Bağlı bileşen analizi: çok küçük izole alanları at
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0

    return mask


def recover_original_gray(bgr_image: np.ndarray, green_mask: np.ndarray) -> np.ndarray:
    """
    Yeşil overlay altındaki orijinal gri görüntüyü kurtarır.

    Mantık:
    - Saf gri piksel: R = G = B = gray_value
    - Yeşil paint sonrası: R ≈ gray_value, G = yüksek, B ≈ gray_value
    - Dolayısıyla yeşil bölgelerde: original ≈ (R + B) / 2

    Args:
        bgr_image  : BGR görüntü
        green_mask : extract_green_mask() çıktısı (0/255)

    Returns:
        gray: uint8 grayscale görüntü, orijinal X-ray'e en yakın hali
    """
    b, g, r = cv2.split(bgr_image)

    # Yeşil olmayan bölgeler: standart grayscale dönüşümü
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Yeşil boyalı bölgeler: G kanalı bozulmuş, R+B ortalamasını kullan
    recovered = ((r.astype(np.int32) + b.astype(np.int32)) // 2).astype(np.uint8)

    painted_pixels = green_mask > 0
    gray[painted_pixels] = recovered[painted_pixels]

    return gray


def process_annotated_image(image_path: str, min_area: int = 100):
    """
    Yeşil boyalı tek bir görüntüyü işler.

    Args:
        image_path : DICOM veya PNG/JPG dosya yolu
        min_area   : Minimum maske bileşen alanı

    Returns:
        gray_image      : Orijinal akciğer grafisi (uint8, HxW)
        binary_mask     : Pnömotoraks maskesi (uint8, HxW, 0/255)
        has_pneumothorax: bool — görüntüde pnömotoraks var mı
    """
    bgr = load_image(image_path)
    mask = extract_green_mask(bgr, min_area=min_area)
    gray = recover_original_gray(bgr, mask)
    has_pneumothorax = bool(mask.max() > 0)
    return gray, mask, has_pneumothorax


def overlay_mask_on_image(
    gray: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    color_bgr: tuple = (0, 60, 220),
) -> np.ndarray:
    """
    Gri görüntü üzerine maskeyi renkli transparan overlay olarak bindirir.
    FastAPI sonuç ekranı ve Grad-CAM karşılaştırması için kullanılır.

    Args:
        gray      : Grayscale X-ray (uint8, HxW)
        mask      : Binary maske (uint8, HxW, 0/255)
        alpha     : Overlay yoğunluğu (0=görünmez, 1=opak)
        color_bgr : Overlay rengi (varsayılan kırmızı)

    Returns:
        result: BGR görüntü
    """
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()
    overlay[mask > 0] = list(color_bgr)
    return cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0)
