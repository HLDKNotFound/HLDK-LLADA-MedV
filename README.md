# 🩺 LLaDA-MedV: Large Language Diffusion Models for Biomedical Image Understanding

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/HLDKNotFound/HLDK-LLADA-MedV/blob/main/notebook/all_datasets_vqa_experiments_molab.py)
[![Paper](https://img.shields.io/badge/arXiv-2508.01617-b31b1b)](https://arxiv.org/abs/2508.01617v1)
[![HuggingFace Model](https://img.shields.io/badge/Model-XZDong123%2FLLaDA--MedV-orange)](https://huggingface.co/XZDong123/LLaDA-MedV)
[![HuggingFace Dataset](https://img.shields.io/badge/Dataset-HLDKNotFound%2Fllada--medv--vqa-blue)](https://huggingface.co/datasets/HLDKNotFound/llada-medv-vqa)
[![Hardware](https://img.shields.io/badge/GPU-RTX%206000%20Blackwell%20(96GB%20%2F%2048GB)-76b900)](https://molab.marimo.io)

Repository nghiên cứu, tối ưu hóa và thực thi suy luận mô hình **LLaDA-MedV** (CVPR 2026) trên 3 benchmark thị giác y sinh lớn: **SLAKE**, **VQA-RAD**, và **PathVQA**. Được tối ưu hóa chuyên sâu cho phần cứng **NVIDIA RTX 6000 Ada / Blackwell 96GB VRAM** trên nền tảng đám mây [molab](https://molab.marimo.io).

---

## 🚀 Điểm Nổi Bật

1. **Khởi Chạy 1-Click Trên Molab Cloud**:
   - Bấm vào badge [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/HLDKNotFound/HLDK-LLADA-MedV/blob/main/notebook/all_datasets_vqa_experiments_molab.py) để mở trực tiếp notebook trên môi trường GPU của Molab.
2. **Tự Động Vá Lỗi Tương Thích `transformers`**:
   - Khắc phục triệt để lỗi gỡ bỏ hàm `find_pruneable_heads_and_indices` trên `transformers >= 4.45` và Python 3.13.
   - Xóa bỏ breakpoint debug `(Pdb)` còn sót lại từ repo gốc.
3. **Tự Động Nạp Dữ Liệu Từ Hugging Face Hub**:
   - Tự động kiểm tra và tải các tập dữ liệu Parquet từ `HLDKNotFound/llada-medv-vqa`.
4. **Hỗ Trợ GPU Blackwell 96GB**:
   - Khai thác băng thông bộ nhớ cực lớn với `batch_size = 64 / 128` song song, nạp đầy đủ weights `XZDong123/LLaDA-MedV` (~16.2 GB).
5. **Xuất 3 File CSV Độc Lập Cho 8,231 Mẫu Kiểm Thử**:
   - `vqa_rad_predictions.csv` (451 mẫu)
   - `slake_predictions.csv` (1,061 mẫu)
   - `pathvqa_predictions.csv` (6,719 mẫu)
   - `all_datasets_benchmark_summary.csv`

---

## 📁 Cấu Trúc Thư Mục

```text
HLDK-LLADA-MedV/
├── HLDK-LLADA-MedV/           # Mã nguồn core kiến trúc LLaDA-MedV (đã fix tương thích)
│   └── llava/
│       ├── model/
│       │   ├── builder.py
│       │   └── multimodal_resampler/qformer.py
│       └── ...
├── notebook/                  # Notebook thử nghiệm
│   ├── all_datasets_vqa_experiments.ipynb      # Bản Jupyter Notebook chuẩn
│   └── all_datasets_vqa_experiments_molab.py   # Bản Marimo Reactive cho Molab
├── scripts/                   # Scripts xây dựng & kiểm thử
│   └── build_all_datasets_notebook.py
├── docs/                      # Tài liệu phân tích paper & metric
└── README.md
```

---

## ⚡ Hướng Dẫn Sử Dụng

### Cách 1: Chạy trực tiếp trên Molab Cloud (Khuyến nghị)
1. Mở link: [https://molab.marimo.io/github/HLDKNotFound/HLDK-LLADA-MedV/blob/main/notebook/all_datasets_vqa_experiments_molab.py](https://molab.marimo.io/github/HLDKNotFound/HLDK-LLADA-MedV/blob/main/notebook/all_datasets_vqa_experiments_molab.py)
2. Chọn GPU môi trường (ví dụ RTX 6000 Blackwell hoặc A100).
3. Chạy tuần tự các Cell để thực hiện đánh giá.

### Cách 2: Chạy trên máy trạm Local / Server
```bash
git clone https://github.com/HLDKNotFound/HLDK-LLADA-MedV.git
cd HLDK-LLADA-MedV
pip install -r HLDK-LLADA-MedV/requirements.txt
jupyter notebook notebook/all_datasets_vqa_experiments.ipynb
```

---

## 📜 Trích Dẫn Bài Báo (Citation)

```bibtex
@article{dong2025llada,
  title={LLaDA-MedV: Exploring Large Language Diffusion Models for Biomedical Image Understanding},
  author={Dong, Xiaozheng and others},
  journal={arXiv preprint arXiv:2508.01617},
  year={2026}
}
```
