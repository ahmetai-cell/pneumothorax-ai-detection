"""
Dataset kararlılık testleri — bozuk/eksik dosya koruma.

Çalıştır:
    pytest tests/test_dataset_stability.py -v
"""

import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import pytest


class TestPneumothoraxDatasetStability:
    """src/preprocessing/dataset.py — SIIM dataset."""

    def _make_df(self, img_id="test_img"):
        return pd.DataFrame([{"image_id": img_id, "EncodedPixels": "-1"}])

    def test_missing_file_raises_not_found(self, tmp_path):
        from src.preprocessing.dataset import PneumothoraxDataset
        df  = self._make_df("nonexistent")
        ds  = PneumothoraxDataset(df, img_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="bozuk veya eksik"):
            _ = ds[0]

    def test_valid_image_loads(self, tmp_path):
        from src.preprocessing.dataset import PneumothoraxDataset
        img = np.zeros((256, 256), dtype=np.uint8)
        img_path = tmp_path / "test_img.png"
        cv2.imwrite(str(img_path), img)

        df  = self._make_df("test_img")
        ds  = PneumothoraxDataset(df, img_dir=str(tmp_path))
        image, mask, label = ds[0]

        assert image.shape == (1, 512, 512), f"Yanlış şekil: {image.shape}"
        assert mask.shape  == (1, 512, 512)
        assert label.item() == 0.0

    def test_corrupted_image_raises(self, tmp_path):
        """Bozuk PNG dosyası FileNotFoundError tetiklemeli."""
        from src.preprocessing.dataset import PneumothoraxDataset
        bad_path = tmp_path / "bad_img.png"
        bad_path.write_bytes(b"not an image at all")

        df  = self._make_df("bad_img")
        ds  = PneumothoraxDataset(df, img_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            _ = ds[0]


class TestPTXDatasetStability:
    """src/preprocessing/ptx_dataset.py — yerel dataset."""

    def _make_df(self, img_path, mask_path=None, label=1):
        return pd.DataFrame([{
            "img_path":  img_path,
            "mask_path": mask_path,
            "label":     label,
            "site":      "SiteA",
        }])

    def test_missing_image_raises(self, tmp_path):
        from src.preprocessing.ptx_dataset import PTXDataset
        df = self._make_df(str(tmp_path / "nofile.png"), str(tmp_path / "nomask.png"))
        ds = PTXDataset(df)
        with pytest.raises(FileNotFoundError):
            _ = ds[0]

    def test_valid_positive_sample(self, tmp_path):
        from src.preprocessing.ptx_dataset import PTXDataset
        img  = np.zeros((256, 256), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:150, 100:150] = 255

        img_path  = tmp_path / "case.png"
        mask_path = tmp_path / "mask.png"
        cv2.imwrite(str(img_path), img)
        cv2.imwrite(str(mask_path), mask)

        df = self._make_df(str(img_path), str(mask_path), label=1)
        ds = PTXDataset(df, img_size=128)
        image_t, mask_t, label_t = ds[0]

        assert image_t.shape == (1, 128, 128)
        assert mask_t.shape  == (1, 128, 128)
        assert label_t.item() == 1.0
        assert mask_t.sum() > 0, "Pozitif vaka maskesi sıfır!"

    def test_negative_sample_zero_mask(self, tmp_path):
        from src.preprocessing.ptx_dataset import PTXDataset
        img = np.full((256, 256), 128, dtype=np.uint8)
        img_path = tmp_path / "neg.png"
        cv2.imwrite(str(img_path), img)

        df = self._make_df(str(img_path), mask_path=None, label=0)
        ds = PTXDataset(df, img_size=128)
        _, mask_t, label_t = ds[0]

        assert label_t.item() == 0.0
        assert mask_t.sum().item() == 0.0, "Negatif vaka maskesi sıfır olmalı!"


class TestReadImageRobustness:
    """_read_image fonksiyonu bozuk girdi için açıklayıcı hata vermeli."""

    def test_nonexistent_png_raises(self, tmp_path):
        from src.preprocessing.ptx_dataset import _read_image
        with pytest.raises(FileNotFoundError, match="Görüntü okunamadı"):
            _read_image(str(tmp_path / "ghost.png"))

    def test_nonexistent_mask_raises(self, tmp_path):
        from src.preprocessing.ptx_dataset import _read_mask
        with pytest.raises(FileNotFoundError, match="Maske okunamadı"):
            _read_mask(str(tmp_path / "ghost_mask.png"))


class TestUploadValidation:
    """api/main.py upload validasyonu."""

    def test_allowed_extensions_pass(self):
        from api.main import _validate_upload
        from unittest.mock import MagicMock

        for ext in [".png", ".jpg", ".jpeg", ".dcm", ".dicom"]:
            mock = MagicMock()
            mock.filename = f"file{ext}"
            result = _validate_upload(mock)
            assert result in ("dicom", "raster")

    def test_unknown_extension_raises_400(self):
        from api.main import _validate_upload
        from fastapi import HTTPException
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.filename = "file.bmp"
        with pytest.raises(HTTPException) as exc_info:
            _validate_upload(mock)
        assert exc_info.value.status_code == 400

    def test_dicom_detected_correctly(self):
        from api.main import _validate_upload
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.filename = "scan.dcm"
        assert _validate_upload(mock) == "dicom"

    def test_png_detected_as_raster(self):
        from api.main import _validate_upload
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.filename = "xray.png"
        assert _validate_upload(mock) == "raster"
