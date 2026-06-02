"""
Normalizasyon tutarlılık testleri.

Çalıştır:
    pytest tests/test_normalization.py -v
"""

import numpy as np
import pytest
import torch


def _albumentations_normalize(img_uint8: np.ndarray) -> np.ndarray:
    """
    Eğitim pipeline'ını simüle eder:
    src/data/transforms.py: A.Normalize(mean=0.485, std=0.229, max_pixel_value=255)
    uint8 girdiye uygulandığında: (pixel/255 - 0.485) / 0.229
    Çıktı aralığı: [-2.12, +2.25]
    """
    return (img_uint8.astype(np.float32) / 255.0 - 0.485) / 0.229


class TestNormalizeImage:
    def setup_method(self):
        from src.utils.normalize import normalize_image
        self.normalize_image = normalize_image

    def test_uint8_input_matches_training(self):
        """uint8 görüntü → normalize_image, eğitim pipeline ile aynı sonucu vermeli."""
        rng = np.random.default_rng(42)
        img_uint8 = rng.integers(0, 256, (512, 512), dtype=np.uint8)

        expected = _albumentations_normalize(img_uint8)
        result   = self.normalize_image(img_uint8)

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5,
            err_msg="uint8 girdi için eğitim normalizasyonuyla uyumsuz")

    def test_float01_input_matches_training(self):
        """[0,1] float girdi, aynı sonucu vermeli (uint8/255 ile aynı)."""
        rng = np.random.default_rng(42)
        img_uint8 = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        img_01    = img_uint8.astype(np.float32) / 255.0

        result_uint8 = self.normalize_image(img_uint8)
        result_float = self.normalize_image(img_01)

        np.testing.assert_allclose(result_uint8, result_float, rtol=1e-5, atol=1e-5,
            err_msg="uint8 ve float01 girdi farklı sonuç veriyor")

    def test_output_range(self):
        """Görüntü çıktısı [-2.12, +2.25] aralığında olmalı."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        result = self.normalize_image(img)
        assert result.min() >= -2.2, f"Min çok düşük: {result.min():.4f}"
        assert result.max() <=  2.3, f"Max çok yüksek: {result.max():.4f}"

    def test_output_dtype(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = self.normalize_image(img)
        assert result.dtype == np.float32

    def test_black_image(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        result = self.normalize_image(img)
        expected_val = (0.0 - 0.485) / 0.229  # -2.1179
        np.testing.assert_allclose(result, expected_val, atol=1e-5)

    def test_white_image(self):
        img = np.full((64, 64), 255, dtype=np.uint8)
        result = self.normalize_image(img)
        expected_val = (1.0 - 0.485) / 0.229  # +2.2489
        np.testing.assert_allclose(result, expected_val, atol=1e-5)


class TestNormalizeToTensor:
    def setup_method(self):
        from src.utils.normalize import normalize_to_tensor
        self.normalize_to_tensor = normalize_to_tensor

    def test_output_shape(self):
        img = np.zeros((512, 512), dtype=np.uint8)
        t = self.normalize_to_tensor(img)
        assert t.shape == (1, 1, 512, 512), f"Beklenen (1,1,512,512), alındı {t.shape}"

    def test_output_dtype(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        t = self.normalize_to_tensor(img)
        assert t.dtype == torch.float32

    def test_device_cpu(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        t = self.normalize_to_tensor(img, device=torch.device("cpu"))
        assert t.device.type == "cpu"

    def test_values_match_normalize_image(self):
        from src.utils.normalize import normalize_image
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        expected = normalize_image(img)
        t = self.normalize_to_tensor(img)
        np.testing.assert_allclose(
            t.squeeze().numpy(), expected, rtol=1e-5, atol=1e-5
        )


class TestNormalizationConsistency:
    """api, tta, predict modülleri aynı formülü kullandığını doğrular."""

    def test_api_uses_normalize_to_tensor(self):
        """api/main.py _inference_sync normalize_to_tensor kullanıyor olmalı."""
        import inspect
        import api.main as api_mod
        src = inspect.getsource(api_mod._inference_sync)
        assert "normalize_to_tensor" in src, \
            "_inference_sync() normalize_to_tensor kullanmıyor — normalizasyon tutarsız!"

    def test_tta_uses_normalize_image(self):
        """tta.py'nin _build_tta_variants() normalize_image import etmeli."""
        import inspect
        import src.utils.tta as tta_mod
        src = inspect.getsource(tta_mod._build_tta_variants)
        assert "normalize_image" in src, \
            "_build_tta_variants() normalize_image kullanmıyor — tutarsızlık!"

    def test_predict_py_uses_normalize_to_tensor(self):
        """predict.py normalize_to_tensor import etmeli."""
        import inspect
        import predict as predict_mod
        src = inspect.getsource(predict_mod.predict)
        assert "normalize_to_tensor" in src, \
            "predict.py normalize_to_tensor kullanmıyor — tutarsızlık!"

    def test_gradcam_result_uses_normalize_to_tensor(self):
        """gradcam.generate_gradcam_result() normalize_to_tensor kullanmalı."""
        import inspect
        import src.utils.gradcam as gc_mod
        src = inspect.getsource(gc_mod.generate_gradcam_result)
        assert "normalize_to_tensor" in src, \
            "generate_gradcam_result() normalize_to_tensor kullanmıyor — tutarsızlık!"
