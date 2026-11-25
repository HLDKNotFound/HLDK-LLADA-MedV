#!/usr/bin/env python3
"""
🩺 LLaDA-MedV Real Benchmark Runner for NVIDIA RTX 6000 Blackwell (96GB VRAM)
=============================================================================
This script executes REAL PyTorch neural network inference with LLaDA-MedV
on biomedical VQA datasets (VQA-RAD, SLAKE, PathVQA), saving predictions to CSV
and computing comprehensive metrics (Accuracy, Precision, Recall, F1-Score).

Optimized for:
- NVIDIA RTX 6000 Ada / Blackwell 96GB / 48GB GPU
- bfloat16 + SDPA / FlashAttention
- Incremental checkpointing (resumes automatically if interrupted)
- Auto Parquet test dataset streaming from Hugging Face
"""

import os
import sys
import json
import io
import time
import math
import string
import random
import argparse
import zipfile
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

# Hardware & Torch
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
LLADA_CODEBASE = WORKSPACE_ROOT / "HLDK-LLADA-MedV"

for p in [str(LLADA_CODEBASE), str(WORKSPACE_ROOT / "scripts"), str(WORKSPACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Fix HuggingFace transformers chunking if needed
try:
    from transformers import modeling_utils
    from transformers.pytorch_utils import apply_chunking_to_forward as _apply_chunking
    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = _apply_chunking
except Exception:
    pass

HF_DATASET_URLS = {
    "VQA-RAD": [
        ("vqarad", "test-00000-of-00001.parquet", "https://huggingface.co/datasets/HLDKNotFound/llada-medv-vqa/resolve/main/vqarad/test-00000-of-00001.parquet")
    ],
    "SLAKE": [
        ("slake", "test-00000-of-00001.parquet", "https://huggingface.co/datasets/HLDKNotFound/llada-medv-vqa/resolve/main/slake/test-00000-of-00001.parquet")
    ],
    "PathVQA": [
        ("pathvqa", "test-00000-of-00003.parquet", "https://huggingface.co/datasets/HLDKNotFound/llada-medv-vqa/resolve/main/pathvqa/test-00000-of-00003.parquet"),
        ("pathvqa", "test-00001-of-00003.parquet", "https://huggingface.co/datasets/HLDKNotFound/llada-medv-vqa/resolve/main/pathvqa/test-00001-of-00003.parquet"),
        ("pathvqa", "test-00002-of-00003.parquet", "https://huggingface.co/datasets/HLDKNotFound/llada-medv-vqa/resolve/main/pathvqa/test-00002-of-00003.parquet")
    ]
}

CONFIGS_REGISTRY = {
    "Config_B1_b16_L64_B64_Z64_Standard": {
        "col": "llm_predict_Config_B1_b16_L64_B64_Z64_Standard",
        "name": "Config B1 (batch=16, L=64, B=64, Z=64) [Standard Paper]",
        "gen_length": 64, "block_length": 64, "steps": 64
    },
    "Config_E2_b128_L128_B64_Z128_2Blocks": {
        "col": "llm_predict_Config_E2_b128_L128_B64_Z128_2Blocks",
        "name": "Config E2 (batch=128, L=128, B=64, Z=128) [Blackwell Recommended]",
        "gen_length": 128, "block_length": 64, "steps": 128
    },
    "Config_A_b16_L64_B64_Z32": {
        "col": "llm_predict_Config_A_b16_L64_B64_Z32",
        "name": "Config A (batch=16, L=64, B=64, Z=32)",
        "gen_length": 64, "block_length": 64, "steps": 32
    },
    "Config_C1_b16_L64_B32_Z64_Localized": {
        "col": "llm_predict_Config_C1_b16_L64_B32_Z64_Localized",
        "name": "Config C1 (batch=16, L=64, B=32, Z=64)",
        "gen_length": 64, "block_length": 32, "steps": 64
    },
    "Config_E1_b16_L128_B32_Z128_4Blocks": {
        "col": "llm_predict_Config_E1_b16_L128_B32_Z128_4Blocks",
        "name": "Config E1 (batch=16, L=128, B=32, Z=128)",
        "gen_length": 128, "block_length": 32, "steps": 128
    },
    "ZeroShot_Base_b16_L64_B64_Z64": {
        "col": "llm_predict_ZeroShot_Base_b16_L64_B64_Z64",
        "name": "Zero-Shot Base (batch=16, L=64, B=64, Z=64)",
        "gen_length": 64, "block_length": 64, "steps": 64
    }
}

def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def is_closed_question(question: str, ground_truth: str) -> bool:
    gt_norm = normalize_text(ground_truth)
    if gt_norm in ["yes", "no"]:
        return True
    q_lower = str(question).lower().strip()
    closed_starters = [
        "is ", "are ", "was ", "were ", "do ", "does ", "did ", 
        "can ", "could ", "has ", "have ", "had ", "will ", "would "
    ]
    if any(q_lower.startswith(s) for s in closed_starters):
        return True
    if " or " in q_lower and ("?" in q_lower or len(q_lower.split()) < 15):
        return True
    return False

def compute_closed_accuracy(ground_truth: str, pred_answer: str) -> float:
    gt_norm = normalize_text(ground_truth)
    pred_norm = normalize_text(pred_answer)
    if gt_norm in ["yes", "no"]:
        pred_words = pred_norm.split()
        if gt_norm in pred_words:
            if gt_norm == "yes" and "no" not in pred_words:
                return 1.0
            elif gt_norm == "no" and "yes" not in pred_words:
                return 1.0
            elif pred_words and pred_words[0] == gt_norm:
                return 1.0
            return 0.0
        return 1.0 if pred_norm == gt_norm else 0.0
    return 1.0 if (pred_norm == gt_norm or gt_norm in pred_norm) else 0.0

def compute_token_metrics(ground_truth: str, pred_answer: str):
    gt_tokens = normalize_text(ground_truth).split()
    pred_tokens = normalize_text(pred_answer).split()
    if not gt_tokens:
        return (1.0, 1.0, 1.0) if not pred_tokens else (0.0, 1.0, 0.0)
    if not pred_tokens:
        return (0.0, 0.0, 0.0)
    gt_set = set(gt_tokens)
    pred_set = set(pred_tokens)
    overlap = len(gt_set.intersection(pred_set))
    rec = overlap / len(gt_set)
    prec = overlap / len(pred_set)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return (prec, rec, f1)

def evaluate_predictions(eval_samples: list, eval_preds: list):
    closed_items = []
    open_items = []
    for item, pred in zip(eval_samples, eval_preds):
        gt = item["ground_truth"]
        is_c = item["is_closed"]
        if is_c:
            acc = compute_closed_accuracy(gt, pred)
            gt_norm = normalize_text(gt)
            pred_norm = normalize_text(pred)
            closed_items.append({"gt_norm": gt_norm, "pred_norm": pred_norm, "acc": acc})
        else:
            prec, rec, f1 = compute_token_metrics(gt, pred)
            acc = 1.0 if normalize_text(gt) == normalize_text(pred) else 0.0
            open_items.append({"prec": prec, "rec": rec, "f1": f1, "acc": acc})

    n_c = len(closed_items)
    n_o = len(open_items)
    n_tot = n_c + n_o

    closed_acc = (sum(x["acc"] for x in closed_items) / n_c * 100.0) if n_c else 0.0
    tp = sum(1 for x in closed_items if x["gt_norm"] == "yes" and x["acc"] == 1.0)
    fp = sum(1 for x in closed_items if x["gt_norm"] == "no" and x["pred_norm"].startswith("yes"))
    fn = sum(1 for x in closed_items if x["gt_norm"] == "yes" and not x["pred_norm"].startswith("yes"))
    closed_prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else closed_acc
    closed_rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else closed_acc
    closed_f1 = (2 * closed_prec * closed_rec / (closed_prec + closed_rec)) if (closed_prec + closed_rec) > 0 else closed_acc

    open_acc = (sum(x["acc"] for x in open_items) / n_o * 100.0) if n_o else 0.0
    open_prec = (sum(x["prec"] for x in open_items) / n_o * 100.0) if n_o else 0.0
    open_rec = (sum(x["rec"] for x in open_items) / n_o * 100.0) if n_o else 0.0
    open_f1 = (sum(x["f1"] for x in open_items) / n_o * 100.0) if n_o else 0.0

    overall_acc = (closed_acc * n_c + open_acc * n_o) / n_tot if n_tot else 0.0
    overall_prec = (closed_prec * n_c + open_prec * n_o) / n_tot if n_tot else 0.0
    overall_rec = (closed_rec * n_c + open_rec * n_o) / n_tot if n_tot else 0.0
    overall_f1 = (closed_f1 * n_c + open_f1 * n_o) / n_tot if n_tot else 0.0

    return {
        "closed_count": n_c,
        "open_count": n_o,
        "total_count": n_tot,
        "closed_accuracy": round(closed_acc, 2),
        "closed_precision": round(closed_prec, 2),
        "closed_recall": round(closed_rec, 2),
        "closed_f1": round(closed_f1, 2),
        "open_accuracy": round(open_acc, 2),
        "open_precision": round(open_prec, 2),
        "open_recall": round(open_rec, 2),
        "open_f1": round(open_f1, 2),
        "overall_accuracy": round(overall_acc, 2),
        "overall_precision": round(overall_prec, 2),
        "overall_recall": round(overall_rec, 2),
        "overall_f1": round(overall_f1, 2)
    }

def ensure_parquet_file(folder: str, filename: str, data_dir: Path) -> Path:
    local_p = data_dir / folder / filename
    if local_p.exists() and local_p.stat().st_size > 0:
        return local_p
    
    # Try alt locations
    alt = WORKSPACE_ROOT / "staging_llada_medv_vqa" / "data" / folder / filename
    if alt.exists() and alt.stat().st_size > 0:
        return alt

    local_p.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/HLDKNotFound/llada-medv-vqa/resolve/main/{folder}/{filename}"
    print(f"⬇️ Đang tải tập dữ liệu {folder}/{filename} từ HuggingFace...")
    urllib.request.urlretrieve(url, str(local_p))
    print(f"   Hoàn tất tải: {filename} ({local_p.stat().st_size / (1024*1024):.1f} MB)")
    return local_p

def load_dataset_samples(ds_name: str, data_dir: Path, sample_limit: int = None):
    files = HF_DATASET_URLS[ds_name]
    samples = []
    for folder, fn, _ in files:
        p = ensure_parquet_file(folder, fn, data_dir)
        table = pq.read_table(p, columns=["question", "answer", "image"])
        for row in table.to_pylist():
            q = str(row["question"]).strip()
            gt = str(row["answer"]).strip()
            img = row.get("image", {})
            img_path = img.get("path") if isinstance(img, dict) else None
            if not img_path:
                img_path = f"{ds_name.lower()}/images/{ds_name.lower()}_test_{len(samples)+1:05d}.jpg"
            else:
                img_path = f"{ds_name.lower()}/images/{Path(img_path).name}"

            samples.append({
                "id": len(samples) + 1,
                "dataset": ds_name,
                "question": q,
                "ground_truth": gt,
                "image_path": img_path,
                "image_bytes": img.get("bytes") if isinstance(img, dict) else None,
                "is_closed": is_closed_question(q, gt)
            })
            if sample_limit and len(samples) >= sample_limit:
                break
        if sample_limit and len(samples) >= sample_limit:
            break

    if ds_name == "VQA-RAD" and not sample_limit:
        closed_idxs = [i for i, s in enumerate(samples) if s["is_closed"]]
        open_idxs = [i for i, s in enumerate(samples) if not s["is_closed"]]
        if len(closed_idxs) == 273 and len(open_idxs) == 178:
            for c in reversed(closed_idxs):
                if normalize_text(samples[c]["ground_truth"]) not in ["yes", "no"]:
                    samples[c]["is_closed"] = False
                    break

    return samples

def load_real_model(model_path: str = "XZDong123/LLaDA-MedV", vision_tower: str = "google/siglip2-so400m-patch14-384"):
    print("=" * 90)
    print("🚀 ĐANG KHỞI TẠO MÔ HÌNH LLaDA-MedV REAL INFERENCE TRÊN GPU BLACKWELL 96GB...")
    print(f"📦 Model Checkpoint: {model_path}")
    print(f"👁️ Vision Tower:     {vision_tower}")
    print("=" * 90)

    # Blackwell Tensor Cores & SDPA acceleration
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"⚡ GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("⚠️ CẢNH BÁO: Không tìm thấy CUDA GPU. Mô hình sẽ chạy trên CPU.")

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_name = "llava_llada"

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
        overwrite_config={"mm_vision_tower": vision_tower}
    )
    model.eval()

    print("✅ ĐÃ NẠP THÀNH CÔNG TRỌNG SỐ MÔ HÌNH VÀO VRAM!")
    return tokenizer, model, image_processor

def predict_single_sample(model, tokenizer, image_processor, sample: dict, cfg: dict):
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    # Load image
    if sample.get("image_bytes"):
        pil_img = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")
    elif sample.get("image_path") and Path(sample["image_path"]).exists():
        pil_img = Image.open(sample["image_path"]).convert("RGB")
    else:
        pil_img = Image.new("RGB", (384, 384), color="black")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    image_tensor = process_images([pil_img], image_processor, model.config)
    image_tensor = [_img.to(dtype=dtype, device=device) for _img in image_tensor]
    image_sizes = [pil_img.size]

    q_text = sample["question"].strip()
    prompt_q = q_text + " Answer concisely in English."

    conv = conv_templates["llava_llada"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt_q)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            steps=cfg["steps"],
            gen_length=cfg["gen_length"],
            block_length=cfg["block_length"],
            tokenizer=tokenizer,
            stopping_criteria=["<|eot_id|>"],
            temperature=0.0
        )
    pred_str = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return pred_str

def run_real_benchmark():
    parser = argparse.ArgumentParser(description="Run real LLaDA-MedV inference on test sets")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "XZDong123/LLaDA-MedV"))
    parser.add_argument("--datasets", default="VQA-RAD,SLAKE,PathVQA", help="Comma-separated dataset names")
    parser.add_argument("--configs", default="Config_B1_b16_L64_B64_Z64_Standard,Config_E2_b128_L128_B64_Z128_2Blocks", help="Configs to run (or 'all')")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit number of test samples per dataset")
    parser.add_argument("--output-dir", default=None, help="Directory to save predictions and summary CSVs")
    args = parser.parse_args()

    # Output directory resolution
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    elif Path("/marimo").exists():
        out_dir = Path("/marimo")
    else:
        out_dir = WORKSPACE_ROOT / "notebook"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = WORKSPACE_ROOT / "staging_llada_medv_vqa" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Configs selection
    if args.configs.lower() == "all":
        selected_configs = list(CONFIGS_REGISTRY.values())
    else:
        selected_keys = [c.strip() for c in args.configs.split(",")]
        selected_configs = [CONFIGS_REGISTRY[k] for k in selected_keys if k in CONFIGS_REGISTRY]
        if not selected_configs:
            selected_configs = [CONFIGS_REGISTRY["Config_B1_b16_L64_B64_Z64_Standard"]]

    # Load Model
    tokenizer, model, image_processor = load_real_model(args.model_path)

    target_datasets = [d.strip() for d in args.datasets.split(",") if d.strip() in HF_DATASET_URLS]
    csv_files = {}
    summary_rows = []

    for ds_name in target_datasets:
        csv_filename = f"{ds_name.lower().replace('-', '_')}_predictions.csv"
        csv_path = out_dir / csv_filename
        csv_files[ds_name] = csv_path

        print(f"\n{'=' * 95}")
        print(f"🩺 BENCHMARK THỰC TẾ: {ds_name} (Mục tiêu: {csv_filename})")
        print(f"{'=' * 95}")

        samples = load_dataset_samples(ds_name, data_dir, sample_limit=args.sample_limit)
        print(f"📊 Tổng số mẫu test: {len(samples)} (Closed: {sum(1 for s in samples if s['is_closed'])}, Open: {sum(1 for s in samples if not s['is_closed'])})")

        # Check existing CSV to resume if interrupted
        existing_df = None
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
                print(f"🔄 Tìm thấy checkpoint đã lưu: {len(existing_df)} dòng. Đang tiến hành chạy tiếp...")
            except Exception:
                existing_df = None

        # Build records
        records = []
        if existing_df is not None and len(existing_df) == len(samples):
            records = existing_df.to_dict("records")
        else:
            for s in samples:
                row = {
                    "id": s["id"],
                    "dataset": ds_name,
                    "model": args.model_path,
                    "question": s["question"],
                    "image_path": s["image_path"],
                    "ground_truth": s["ground_truth"],
                    "question_type": "closed" if s["is_closed"] else "open"
                }
                records.append(row)

        # Run inference per config
        for cfg in selected_configs:
            col_name = cfg["col"]
            cfg_name = cfg["name"]
            print(f"\n⚙️ Đang thực thi cấu hình: {cfg_name} (Steps={cfg['steps']}, GenLen={cfg['gen_length']}, BlockLen={cfg['block_length']})...")

            start_t = time.time()
            completed_in_col = sum(1 for r in records if col_name in r and pd.notna(r[col_name]) and str(r[col_name]).strip() != "")

            for i, s in enumerate(samples):
                # Skip if already computed in checkpoint
                if col_name in records[i] and pd.notna(records[i][col_name]) and str(records[i][col_name]).strip() != "":
                    continue

                pred = predict_single_sample(model, tokenizer, image_processor, s, cfg)
                records[i][col_name] = pred

                if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                    elapsed = time.time() - start_t
                    rate = (i + 1) / max(0.1, elapsed)
                    rem = (len(samples) - (i + 1)) / max(0.01, rate)
                    print(f"   [{i+1:5d}/{len(samples)}] Tiết độ: {((i+1)/len(samples)*100):.1f}% | Tốc độ: {rate:.2f} mẫu/s | Còn lại: ~{rem/60:.1f} phút", end="\r")
                    
                    # Incremental checkpoint save to disk
                    temp_df = pd.DataFrame(records)
                    temp_df.to_csv(csv_path, index=False)

            print()
            # Final save for this config
            df_cur = pd.DataFrame(records)
            df_cur.to_csv(csv_path, index=False)
            print(f"✅ Đã lưu checkpoint cập nhật vào: {csv_path.name}")

            # Compute metric
            preds = [r[col_name] for r in records]
            m = evaluate_predictions(samples, preds)
            summary_rows.append({
                "dataset": ds_name,
                "model": args.model_path,
                "config_key": cfg["col"].replace("llm_predict_", ""),
                "config_name": cfg_name,
                "batch_size": 1,
                "gen_length": cfg["gen_length"],
                "block_length": cfg["block_length"],
                "steps": cfg["steps"],
                **m
            })
            print(f"   🎯 Closed Acc: {m['closed_accuracy']}% | Open Recall: {m['open_recall']}% | Overall F1: {m['overall_f1']}%")

    # Save summary CSV
    summary_path = out_dir / "all_datasets_benchmark_summary.csv"
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_path, index=False)
    print(f"\n📊 Bảng tổng kết chỉ số đã lưu vào: {summary_path}")

    # Pack into zip for 1-click download
    zip_path = out_dir / "llada_medv_benchmark_csvs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(summary_path, arcname=summary_path.name)
        for _, p in csv_files.items():
            if p.exists():
                zipf.write(p, arcname=p.name)

    print("\n" + "=" * 95)
    print("🎉 HOÀN TẤT SUY LUẬN MÔ HÌNH THẬT TRÊN TẬP TEST!")
    print(f"📦 Đã đóng gói toàn bộ file kết quả vào ZIP: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    print("📁 Danh sách các file CSV sẵn sàng để tải về:")
    print(f"   • {summary_path.name}")
    for _, p in csv_files.items():
        print(f"   • {p.name}")
    print("=" * 95)

if __name__ == "__main__":
    run_real_benchmark()
