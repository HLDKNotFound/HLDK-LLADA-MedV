#!/usr/bin/env python3
"""
📥 LLaDA-MedV & SigLIP2 Local Model Downloader
=============================================
Downloads complete Hugging Face model checkpoints to local disk:
1. XZDong123/LLaDA-MedV (~16GB)
2. google/siglip2-so400m-patch14-384 (~1.7GB)

Uses hf_transfer for high-speed parallel downloads.
"""

import os
import sys
import time
from pathlib import Path

# Enable high-speed Rust-based hf-transfer if available
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = WORKSPACE_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Read token from .env if present
token = None
env_file = WORKSPACE_ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            token = line.split("=", 1)[1].strip()
            break

from huggingface_hub import snapshot_download

TARGETS = [
    {
        "repo_id": "google/siglip2-so400m-patch14-384",
        "local_dir": MODELS_DIR / "siglip2-so400m-patch14-384",
        "desc": "Vision Tower: SigLIP2 (SO400M Patch14 384) ~1.7GB"
    },
    {
        "repo_id": "XZDong123/LLaDA-MedV",
        "local_dir": MODELS_DIR / "LLaDA-MedV",
        "desc": "Main Model: LLaDA-MedV (8B Diffusion VLM) ~16GB"
    }
]

def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU TẢI TRỌNG SỐ MÔ HÌNH VỀ MÁY LOCAL...")
    print(f"📁 Thư mục lưu trữ: {MODELS_DIR}")
    print("=" * 80)

    for item in TARGETS:
        repo_id = item["repo_id"]
        local_dir = item["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📦 Đang tải: {item['desc']}...")
        print(f"   Repo ID:   {repo_id}")
        print(f"   Lưu tại:   {local_dir}")

        start_t = time.time()
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=token,
                max_workers=8,
                resume_download=True
            )
            elapsed = time.time() - start_t
            # Calculate total size
            total_bytes = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
            print(f"✅ Hoàn tất tải {repo_id} ({total_bytes / (1024**3):.2f} GB) trong {elapsed:.1f}s!")
        except Exception as e:
            print(f"❌ Lỗi khi tải {repo_id}: {e}")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("🎉 TẤT CẢ MODEL ĐÃ ĐƯỢC TẢI THÀNH CÔNG VỀ LOCAL!")
    print(f"📁 LLaDA-MedV: {MODELS_DIR / 'LLaDA-MedV'}")
    print(f"📁 SigLIP2:    {MODELS_DIR / 'siglip2-so400m-patch14-384'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
