"""
Colab setup: dizinler + .dcm_dir + manifest
python scripts/colab_setup.py

TÜBİTAK 2209-A | Ahmet Demir, Erkan Koçulu
"""
import os, glob, shutil, sys

DRIVE_BASE = '/content/drive/MyDrive/tubitak_pneumothorax'
DRIVE_SIIM = f'{DRIVE_BASE}/data/siim'
DRIVE_CKPT = f'{DRIVE_BASE}/checkpoints'
SIIM_LOCAL = 'data/raw/global/siim'

for d in [SIIM_LOCAL, 'data/processed', 'data/masks/siim', DRIVE_CKPT]:
    os.makedirs(d, exist_ok=True)

# train-rle.csv
rle_dst = f'{SIIM_LOCAL}/train-rle.csv'
if not os.path.exists(rle_dst):
    candidates = (
        [f'{DRIVE_SIIM}/pneumothorax/train-rle.csv', f'{DRIVE_SIIM}/stage_2_train.csv']
        + glob.glob(f'{DRIVE_SIIM}/pneumothorax/*.csv')
        + glob.glob(f'{DRIVE_SIIM}/*.csv')
    )
    found = [c for c in candidates if os.path.exists(c)]
    if found:
        shutil.copy(found[0], rle_dst)
        print(f'RLE CSV kopyalandi: {os.path.basename(found[0])}')
    else:
        print('[!] train-rle.csv bulunamadi'); sys.exit(1)
else:
    print(f'train-rle.csv mevcut')

# .dcm_dir
drive_dcm = f'{DRIVE_SIIM}/dicom-images-train'
with open(f'{SIIM_LOCAL}/.dcm_dir', 'w') as f:
    f.write(drive_dcm + '\n')
print(f'.dcm_dir yazildi -> {drive_dcm}')
print(f'Dizin var: {os.path.isdir(drive_dcm)}')
