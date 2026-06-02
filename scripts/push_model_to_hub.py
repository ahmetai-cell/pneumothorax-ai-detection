"""
Model Versiyonlama — Hugging Face Hub
======================================
Eğitim tamamlandığında checkpoint'leri HF Hub'a yükler.
Git'e girmeyecek büyük dosyalar için versiyon kontrolü sağlar.

Kullanım:
    python scripts/push_model_to_hub.py --version v1.0-ptx498-nih550
    python scripts/push_model_to_hub.py --version v2.0-siim-pretrain --checkpoint results/checkpoints/

Gereksinim:
    pip install huggingface_hub
    HF_TOKEN ve HF_REPO_ID .env'de tanımlı olmalı

TÜBİTAK 2209-A | Ahmet Demir, Erkan Koçulu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent


def push_checkpoints(
    version: str,
    checkpoint_dir: Path,
    repo_id: str,
    token: str,
    commit_message: str | None = None,
) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("huggingface_hub kurulu değil: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    # Repo oluştur (zaten varsa atla)
    try:
        create_repo(repo_id, repo_type="model", private=True, token=token, exist_ok=True)
    except Exception as e:
        print(f"Repo oluşturma uyarısı (devam ediliyor): {e}")

    # Checkpoint dosyalarını topla
    ckpt_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))
    if not ckpt_files:
        print(f"Checkpoint bulunamadı: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nYüklenecek dosyalar ({len(ckpt_files)}):")
    for f in ckpt_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

    # Versiyon metadata'sı
    metadata = {
        "version": version,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "files": [f.name for f in ckpt_files],
        "repo": str(checkpoint_dir),
    }

    # Metadata JSON yükle
    meta_path = checkpoint_dir / "version_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    ckpt_files.append(meta_path)

    msg = commit_message or f"Model {version} — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

    for ckpt in ckpt_files:
        path_in_repo = f"{version}/{ckpt.name}"
        print(f"  Upload: {ckpt.name} → {path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=msg,
        )

    print(f"\n✓ {len(ckpt_files)} dosya yüklendi: https://huggingface.co/{repo_id}/tree/main/{version}")


def pull_checkpoints(
    version: str,
    dest_dir: Path,
    repo_id: str,
    token: str,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub kurulu değil: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
    version_files = [f for f in files if f.startswith(f"{version}/")]

    if not version_files:
        print(f"Versiyon bulunamadı: {version} @ {repo_id}", file=sys.stderr)
        sys.exit(1)

    print(f"\nİndiriliyor: {version} ({len(version_files)} dosya)")
    for file_path in version_files:
        filename = Path(file_path).name
        dest = dest_dir / filename
        print(f"  {file_path} → {dest}")
        api.hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            local_dir=str(dest_dir),
            token=token,
        )
    print(f"\n✓ İndirildi: {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="PTX model versiyonlama — HuggingFace Hub")
    parser.add_argument("--action", choices=["push", "pull"], default="push")
    parser.add_argument("--version", required=True, help="örn: v1.0-ptx498-nih550")
    parser.add_argument("--checkpoint", default=str(ROOT / "results" / "checkpoints"))
    parser.add_argument("--dest",       default=str(ROOT / "results" / "checkpoints"))
    args = parser.parse_args()

    token   = os.getenv("HF_TOKEN", "")
    repo_id = os.getenv("HF_REPO_ID", "")

    if not token:
        print("HF_TOKEN ortam değişkeni tanımlı değil.", file=sys.stderr)
        sys.exit(1)
    if not repo_id:
        print("HF_REPO_ID ortam değişkeni tanımlı değil.", file=sys.stderr)
        sys.exit(1)

    if args.action == "push":
        push_checkpoints(
            version=args.version,
            checkpoint_dir=Path(args.checkpoint),
            repo_id=repo_id,
            token=token,
        )
    else:
        pull_checkpoints(
            version=args.version,
            dest_dir=Path(args.dest),
            repo_id=repo_id,
            token=token,
        )


if __name__ == "__main__":
    main()
