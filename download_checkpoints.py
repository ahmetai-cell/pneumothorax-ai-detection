"""
Model checkpoint'lerini GitHub Releases'tan indirir.
Kullanim: python download_checkpoints.py
"""
import os
import urllib.request
from pathlib import Path

REPO    = "ahmetai-cell/pneumothorax-ai-detection"
RELEASE = "v1.0"
FOLDS   = 5
DEST    = Path("results/checkpoints")

def download(url: str, dest: Path):
    print(f"  Indiriliyor: {dest.name}...", end=" ", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    size = dest.stat().st_size / (1024 * 1024)
    print(f"{size:.0f} MB")

if __name__ == "__main__":
    print("Pnomotoraks AI - Model Checkpointleri")
    print("=" * 40)

    for i in range(1, FOLDS + 1):
        fname = f"fold_{i}_best.pth"
        dest  = DEST / fname
        if dest.exists():
            print(f"  {fname} zaten var, atlaniyor.")
            continue
        url = f"https://github.com/{REPO}/releases/download/{RELEASE}/{fname}"
        download(url, dest)

    print()
    print("Tamamlandi! Uygulamayi baslatmak icin:")
    print("  uvicorn api.main:app --port 8000")
