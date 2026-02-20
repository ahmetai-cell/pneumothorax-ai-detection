"""
PyTorch Dataset — SIIM-ACR Pneumothorax
TÜBİTAK 2209-A | Ahmet Demir
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def rle_decode(mask_rle, shape=(1024, 1024)):
    """Run-length encoding → binary mask"""
    if mask_rle == "-1" or pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


class PneumothoraxDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, img_size=512):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"] + ".png")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0

        mask = rle_decode(row.get("EncodedPixels", "-1"))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 0).astype(np.float32)

        label = 1.0 if mask.sum() > 0 else 0.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        label = torch.tensor(label)

        return image, mask, label
