import os
import shutil
from pathlib import Path
import dotenv
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset

def main():
    env_path = Path("/home/mercury/HLDK-LLADA-MedV/.env")
    env = dotenv.dotenv_values(env_path)
    hf_token = env.get("HF_TOKEN")
    hf_un = env.get("HF_UN", "HLDKNotFound")

    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env!")

    repo_id = f"{hf_un}/llada-medv-vqa"
    print(f"Target Hugging Face Dataset: {repo_id}")

    staging_dir = Path("/home/mercury/HLDK-LLADA-MedV/staging_llada_medv_vqa")
    data_dir = staging_dir / "data"

    slake_dir = data_dir / "slake"
    pathvqa_dir = data_dir / "pathvqa"
    vqarad_dir = data_dir / "vqarad"

    for d in [slake_dir, pathvqa_dir, vqarad_dir]:
        d.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=hf_token)

    # 1. SLAKE (mdwiratathya/SLAKE-vqa-english)
    print("\n--- Downloading and preparing SLAKE ---")
    slake_files = [
        ("data/train-00000-of-00002.parquet", "train-00000-of-00002.parquet"),
        ("data/train-00001-of-00002.parquet", "train-00001-of-00002.parquet"),
        ("data/test-00000-of-00001.parquet", "test-00000-of-00001.parquet"),
        ("data/validation-00000-of-00001.parquet", "validation-00000-of-00001.parquet"),
    ]
    for src_file, dst_name in slake_files:
        dst_path = slake_dir / dst_name
        if not dst_path.exists():
            print(f"Fetching {src_file} -> {dst_name}...")
            cached = hf_hub_download("mdwiratathya/SLAKE-vqa-english", src_file, repo_type="dataset", token=hf_token)
            shutil.copyfile(cached, dst_path)
        print(f"SLAKE file ready: {dst_name} ({dst_path.stat().st_size / (1024*1024):.2f} MB)")

    # 2. VQA-RAD (flaviagiammarino/vqa-rad)
    print("\n--- Downloading and preparing VQA-RAD ---")
    vqarad_files = [
        ("data/train-00000-of-00001-eb8844602202be60.parquet", "train-00000-of-00001.parquet"),
        ("data/test-00000-of-00001-e5bc3d208bb4deeb.parquet", "test-00000-of-00001.parquet"),
    ]
    for src_file, dst_name in vqarad_files:
        dst_path = vqarad_dir / dst_name
        if not dst_path.exists():
            print(f"Fetching {src_file} -> {dst_name}...")
            cached = hf_hub_download("flaviagiammarino/vqa-rad", src_file, repo_type="dataset", token=hf_token)
            shutil.copyfile(cached, dst_path)
        print(f"VQA-RAD file ready: {dst_name} ({dst_path.stat().st_size / (1024*1024):.2f} MB)")

    # 3. PathVQA (flaviagiammarino/path-vqa)
    print("\n--- Downloading and preparing PathVQA ---")
    pathvqa_repo_files = api.list_repo_files("flaviagiammarino/path-vqa", repo_type="dataset")
    pathvqa_parquet_files = sorted([f for f in pathvqa_repo_files if f.endswith(".parquet") and f.startswith("data/")])
    for src_file in pathvqa_parquet_files:
        # e.g. data/train-00000-of-00007-f2d0e9ef9f022d38.parquet -> train-00000-of-00007.parquet
        filename = Path(src_file).name
        parts = filename.split("-")
        # Keep split, index, 'of', total
        # train-00000-of-00007-hash.parquet -> train-00000-of-00007.parquet
        if len(parts) >= 4:
            dst_name = f"{parts[0]}-{parts[1]}-{parts[2]}-{parts[3]}.parquet"
        else:
            dst_name = filename
        dst_path = pathvqa_dir / dst_name
        if not dst_path.exists():
            print(f"Fetching {src_file} -> {dst_name}...")
            cached = hf_hub_download("flaviagiammarino/path-vqa", src_file, repo_type="dataset", token=hf_token)
            shutil.copyfile(cached, dst_path)
        print(f"PathVQA file ready: {dst_name} ({dst_path.stat().st_size / (1024*1024):.2f} MB)")

    # 4. Generate README.md with YAML metadata
    print("\n--- Generating dataset card README.md ---")
    readme_content = """---
annotations_creators:
- expert-generated
language_creators:
- expert-generated
language:
- en
license: cc-by-4.0
task_categories:
- visual-question-answering
task_ids: []
pretty_name: LLaDA-MedV Benchmark Datasets (SLAKE, PathVQA, VQA-RAD)
size_categories:
- 10K<n<100K
tags:
- medical
- vqa
- radiology
- pathology
- multimodal
- llada-medv
configs:
- config_name: slake
  data_files:
  - split: train
    path: data/slake/train-*
  - split: test
    path: data/slake/test-*
  - split: validation
    path: data/slake/validation-*
- config_name: pathvqa
  data_files:
  - split: train
    path: data/pathvqa/train-*
  - split: test
    path: data/pathvqa/test-*
  - split: validation
    path: data/pathvqa/validation-*
- config_name: vqarad
  data_files:
  - split: train
    path: data/vqarad/train-*
  - split: test
    path: data/vqarad/test-*
---

# LLaDA-MedV Medical Visual Question Answering (VQA) Benchmark Datasets

This repository consolidates the 3 primary medical visual question answering (Med-VQA) downstream benchmark datasets evaluated in the paper:
**[LLaDA-MedV: Exploring Large Language Diffusion Models for Biomedical Image Understanding](https://arxiv.org/abs/2508.01617)** (CVPR 2026).

## Subsets Overview

The dataset provides 3 configurations (`slake`, `pathvqa`, `vqarad`) with official train, validation, and test splits:

| Subset | Modality / Domain | Train Examples | Validation Examples | Test Examples | Total Examples |
|---|---|---|---|---|---|
| **`slake`** | Radiology (CT, MRI, X-Ray) | 4,919 | 1,053 | 1,061 | 7,033 |
| **`pathvqa`** | Pathology / Histology | 19,654 | 6,259 | 6,719 | 32,632 |
| **`vqarad`** | Radiology (Head, Chest, Abdomen) | 3,064 | — | 451 | 3,515 |
| **Total** | | **27,637** | **7,312** | **8,231** | **43,180** |

## Data Fields

Each subset shares the standard Hugging Face dataset format:
- `image`: PIL Image object (`Image(mode=None, decode=True)`)
- `question`: `string`, the clinical question regarding the medical image
- `answer`: `string`, the ground-truth answer

## Quick Start / How to Use

Load each benchmark subset effortlessly via Hugging Face `datasets`:

```python
from datasets import load_dataset

# 1. Load SLAKE (Radiology VQA)
slake_train = load_dataset("HLDKNotFound/llada-medv-vqa", "slake", split="train")
slake_test = load_dataset("HLDKNotFound/llada-medv-vqa", "slake", split="test")
print("SLAKE sample:", slake_train[0]["question"], "->", slake_train[0]["answer"])

# 2. Load PathVQA (Pathology VQA)
pathvqa_train = load_dataset("HLDKNotFound/llada-medv-vqa", "pathvqa", split="train")
pathvqa_test = load_dataset("HLDKNotFound/llada-medv-vqa", "pathvqa", split="test")
print("PathVQA sample:", pathvqa_train[0]["question"], "->", pathvqa_train[0]["answer"])

# 3. Load VQA-RAD (Radiology VQA)
vqarad_train = load_dataset("HLDKNotFound/llada-medv-vqa", "vqarad", split="train")
vqarad_test = load_dataset("HLDKNotFound/llada-medv-vqa", "vqarad", split="test")
print("VQA-RAD sample:", vqarad_train[0]["question"], "->", vqarad_train[0]["answer"])
```

## Citations & References

If you use these datasets or LLaDA-MedV, please cite the corresponding works:

```bibtex
@article{dong2025llada,
  title={LLaDA-MedV: Exploring Large Language Diffusion Models for Biomedical Image Understanding},
  author={Dong, Xuanzhao and Zhu, Wenhui and Chen, Xiwen and Wang, Zhipeng and Qiu, Peijie and Tang, Shao and Li, Xin and Wang, Yalin},
  journal={arXiv preprint arXiv:2508.01617},
  year={2025}
}

@article{liu2021slake,
  title={Slake: A semantically-labeled knowledge-enhanced dataset for medical visual question answering},
  author={Liu, Bo and Zhan, Li-Ming and Xu, Ling-Xiang and Ma, Lin and Yang, Yan and Wu, Xiao-Ming},
  journal={arXiv preprint arXiv:2102.09542},
  year={2021}
}

@article{he2020pathvqa,
  title={PathVQA: 30000+ Questions for Medical Visual Question Answering},
  author={He, Xuehai and Zhang, Yichen and Mou, Liyuan and Xing, Eric and Xie, Pengtao},
  journal={arXiv preprint arXiv:2003.10286},
  year={2020}
}

@article{lau2018dataset,
  title={A dataset of clinically generated visual questions and answers about radiology images},
  author={Lau, Jason J and Gayen, Soumya and Kovaleva, Alisa and Demner-Fushman, Dina},
  journal={Scientific Data},
  volume={5},
  number={1},
  pages={1--10},
  year={2018},
  publisher={Nature Publishing Group}
}
```
"""
    readme_path = staging_dir / "README.md"
    readme_path.write_text(readme_content.strip() + "\n", encoding="utf-8")
    print(f"Created {readme_path}")

    # 5. Create / Verify repo on Hugging Face
    print(f"\n--- Ensuring Hugging Face repository exists: {repo_id} ---")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    print("Repository created / verified successfully.")

    # 6. Upload folder
    print(f"\n--- Uploading staging folder to Hugging Face: {repo_id} ---")
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload LLaDA-MedV VQA datasets (SLAKE, PathVQA, VQA-RAD)",
    )
    print("Upload completed successfully!")

    # 7. Verification: load subsets from HF
    print("\n--- Verifying uploaded datasets from Hugging Face ---")
    for cfg in ["slake", "vqarad", "pathvqa"]:
        print(f"\nTesting load_dataset('{repo_id}', '{cfg}')...")
        ds = load_dataset(repo_id, cfg, token=hf_token)
        print(f"Loaded config '{cfg}': splits = {list(ds.keys())}")
        for split_name in ds.keys():
            print(f"  - {split_name}: {len(ds[split_name])} examples")
        sample = ds["test"][0]
        print(f"  Sample test: Q: {sample['question']} | A: {sample['answer']} | Image size: {sample['image'].size}")

    print("\nALL TASKS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
