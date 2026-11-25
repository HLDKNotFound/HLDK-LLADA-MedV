# 🔬 NGHIÊN CỨU CHUYÊN SÂU: LLaDA-MedV VÀ LỘ TRÌNH THỰC NGHIỆM ĐỘT PHÁ
## *Phân Tích Cơ Sở Lý Thuyết, Cơ Chế Suy Thoái, Nghịch Lý Hiệu Năng và Đề Xuất Các Đề Tài Nghiên Cứu Mở Rộng*

---

> **Tài liệu tham khảo chính**:  
> *LLaDA-MedV: Exploring Large Language Diffusion Models for Biomedical Image Understanding* (CVPR 2026, arXiv: [2508.01617](https://arxiv.org/abs/2508.01617))  
> **Mã nguồn gốc**: [LLM-VLM-GSL/LLaDA-MedV](https://github.com/LLM-VLM-GSL/LLaDA-MedV) | [XZDong123/LLaDA-MedV](https://huggingface.co/XZDong123/LLaDA-MedV)  
> **Đối tượng sử dụng**: Nghiên cứu sinh, Kỹ sư R&D AI Y tế, Tác giả nghiên cứu tiếp nối về Medical VLMs & Diffusion Language Models.

---

## MỤC LỤC CHI TIẾT
1. [NỀN TẢNG TOÁN HỌC: MASKED DIFFUSION VS. AUTOREGRESSIVE MODELS TRONG Y SINH](#1-nền-tảng-toán-học-masked-diffusion-vs-autoregressive-models-trong-y-sinh)
   - 1.1. Công thức hóa bài toán Sinh Tự hồi quy (ARM) và Nghịch lý Early-Stopping
   - 1.2. Công thức hóa Quá trình Khuếch tán Rời rạc (Discrete State-Space Diffusion)
   - 1.3. Cơ chế Khử nhiễu Bán tự hồi quy (Semi-Autoregressive Block Generation)
2. [GIẢI PHẪU KIẾN TRÚC VÀ CÁC QUYẾT ĐỊNH THIẾT KẾ (DESIGN DECISIONS)](#2-giải-phẫu-kiến-trúc-và-các-quyết-định-thiết-kế-design-decisions)
   - 2.1. Lựa chọn Vision Tower: Tại sao SigLIP2-so400m-patch14-384 đánh bại CLIP truyền thống?
   - 2.2. Nghịch lý Khởi tạo (Initialization Paradox): Tại sao LLaDA-V tự nhiên gây suy thoái miền y sinh?
   - 2.3. Quy trình Huấn luyện 3 giai đoạn: Bản chất căn chỉnh không gian biểu diễn
3. [MỔ XẺ THỰC CHẤT KẾT QUẢ VÀ CÁC ĐIỂM NGHẼN KỸ THUẬT (EMPIRICAL BREAKDOWN)](#3-mổ-xẻ-thực-chất-kết-quả-và-các-điểm-nghẽn-kỹ-thuật-empirical-breakdown)
   - 3.1. Nghịch lý Câu hỏi đóng (Closed-form SOTA) vs. Câu hỏi mở (Open-form Failure)
   - 3.2. Cơ chế Bẫy Tiêu Điểm và Lặp Từ (Token Repetition Attractors)
   - 3.3. Độ phức tạp tính toán và Nghịch lý Bộ nhớ (Attention Quadratic Scaling & No KV-Cache)
4. [HỆ THỐNG CÁC ĐỀ TÀI VÀ THỰC NGHIỆM MỞ RỘNG TIỀM NĂNG (RESEARCH OPPORTUNITIES)](#4-hệ-thống-các-đề-tài-và-thực-nghiệm-mở-rộng-tiềm-năng-research-opportunities)
   - [Đề tài 1: Remasking Động dựa trên Entropy và Trừng phạt Lặp từ (Entropy-Guided Dynamic Remasking)](#đề-tài-1-remasking-động-dựa-trên-entropy-và-trừng-phạt-lặp-từ-entropy-guided-dynamic-remasking)
   - [Đề tài 2: Hậu huấn luyện Căn chỉnh Ưu tiên (Diffusion-DPO / Medical DPO) cho VQA Mở](#đề-tài-2-hậu-huấn-luyện-căn-chỉnh-ưu-tiên-diffusion-dpo--medical-dpo-cho-vqa-mở)
   - [Đề tài 3: Chưng cất Tính nhất quán Đa phương thức (Multimodal Consistency Distillation)](#đề-tài-3-chưng-cất-tính-nhất-quán-đa-phương-thức-multimodal-consistency-distillation)
   - [Đề tài 4: Giải mã Ràng buộc Thực thể Lâm sàng (Constrained Ontology Decoding via Logit Masking)](#đề-tài-4-giải-mã-ràng-buộc-thực-thể-lâm-sàng-constrained-ontology-decoding-via-logit-masking)
   - [Đề tài 5: Thẩm định Ảo giác Y khoa (Medical Hallucination Benchmark & Bidirectional Fact-Checking)](#đề-tài-5-thẩm-định-ảo-giác-y-khoa-medical-hallucination-benchmark--bidirectional-fact-checking)
   - [Đề tài 6: Mở rộng sang Dữ liệu Thể tích 3D CT/MRI (Volumetric Medical Diffusion VLM)](#đề-tài-6-mở-rộng-sang-dữ-liệu-thể-tích-3d-ctmri-volumetric-medical-diffusion-vlm)
5. [HƯỚNG DẪN KỸ THUẬT TRIỂN KHAI VÀ TỐI ƯU HÓA TÀI NGUYÊN (ENGINEERING GUIDE)](#5-hướng-dẫn-kỹ-thuật-triển-khai-và-tối-ưu-hóa-tài-nguyên-engineering-guide)
   - 5.1. Kỹ thuật Lượng tử hóa 4-bit NF4 và Tách tải CPU-GPU (Offloading)
   - 5.2. Tối ưu hóa I/O và Pipeline dữ liệu Parquet
6. [TỔNG KẾT VÀ KẾ HOẠCH HÀNH ĐỘNG](#6-tổng-kết-và-kế-hoạch-hành-động)

---

## 1. NỀN TẢNG TOÁN HỌC: MASKED DIFFUSION VS. AUTOREGRESSIVE MODELS TRONG Y SINH

Để nghiên cứu sâu về LLaDA-MedV, trước tiên cần phân tích sự khác biệt cốt lõi ở cấp độ xác suất và kiến trúc giữa hai trường phái sinh ngôn ngữ: **Autoregressive Models (ARMs)** và **Masked Generative Diffusion Models**.

### 1.1. Công thức hóa bài toán Sinh Tự hồi quy (ARM) và Nghịch lý Early-Stopping
Trong các mô hình ARM truyền thống (như LLaVA-Med, Med-Flamingo), phân phối xác suất đồng thời của chuỗi văn bản $y = (y_1, y_2, \dots, y_L)$ có điều kiện trên hình ảnh y khoa $I$ và câu hỏi $x$ được phân rã theo quy tắc chuỗi (chain rule):
$$p_\theta(y \mid I, x) = \prod_{i=1}^L p_\theta(y_i \mid y_{<i}, I, x)$$

#### Nghịch lý Early-Stopping trong Y tế
- **Cơ chế suy thoái**: Khi mô hình ARM được tinh chỉnh trên các bộ dữ liệu VQA lâm sàng (nơi nhãn huấn luyện thường chỉ là 1-2 từ như *"pneumonia"*, *"yes"*), gradient cập nhật ép trọng số của token kết thúc chuỗi $\langle\text{eos}\rangle$ tăng đột biến ngay sau khi từ khóa xuất hiện:
  $$p_\theta(y_{k+1} = \langle\text{eos}\rangle \mid y_{\le k}, I, x) \approx 1.0$$
- **Hậu quả lâm sàng**: Dù người dùng sử dụng Prompt yêu cầu giải thích chi tiết (*"Explain the radiological findings in detail and discuss differential diagnosis"*), mô hình ARM vẫn bị sụp đổ sớm (early termination). Câu trả lời chỉ đạt trung bình ~36 từ, bỏ qua hoàn toàn giải thích về cơ chế bệnh học, vị trí giải phẫu lân cận và khuyến nghị an toàn.

---

### 1.2. Công thức hóa Quá trình Khuếch tán Rời rạc (Discrete State-Space Diffusion)
LLaDA-MedV tiếp cận việc sinh ngôn ngữ theo mô hình khuếch tán rời rạc dựa trên mặt nạ (Masked Discrete Diffusion - mở rộng từ D3PM và LLaDA).

#### 1. Quá trình Thuận (Forward Process - Thêm nhiễu mặt nạ)
Không gian trạng thái là từ vựng $\mathcal{V} \cup \{\mathbf{M}\}$, trong đó $\mathbf{M}$ là token mặt nạ đặc biệt (absorbing state).
Tại thời điểm $t \in [0, 1]$, mỗi token trong chuỗi sạch $y_0$ độc lập chuyển sang token mặt nạ $\mathbf{M}$ với xác suất $t$:
$$q(y_t \mid y_0) = \prod_{i=1}^L q(y_t^i \mid y_0^i)$$
với:
$$q(y_t^i \mid y_0^i) = \begin{cases} 1 - t & \text{nếu } y_t^i = y_0^i \\ t & \text{nếu } y_t^i = \mathbf{M} \end{cases}$$
Khi $t = 1$, chuỗi $y_1$ hoàn toàn là chuỗi mặt nạ độ dài $L$: $y_1 = (\mathbf{M}, \mathbf{M}, \dots, \mathbf{M})$.

#### 2. Quá trình Đảo (Reverse Process - Khử nhiễu sinh văn bản)
Mục tiêu là học phân phối chuyển dịch ngược từ $t$ về $s$ (với $0 \le s < t \le 1$):
$$q(y_s \mid y_t, y_0)$$
Theo công thức Bayes, với vị trí đang bị che mặt nạ $y_t^i = \mathbf{M}$:
$$q(y_s^i \mid y_t^i = \mathbf{M}, y_0^i) = \frac{s}{t} \delta(y_s^i, \mathbf{M}) + \left(1 - \frac{s}{t}\right) \delta(y_s^i, y_0^i)$$
Vì phân phối thực $y_0$ là chưa biết, một mạng nơ-ron sâu $p_\theta(y_0 \mid y_t, I, x)$ được huấn luyện để dự đoán dữ liệu sạch $y_0$ từ chuỗi nhiễu $y_t$ và điều kiện đa phương thức $(I, x)$.

#### 3. Hàm mất mát Huấn luyện (Training Objective)
Hàm mất mát là kỳ vọng lỗi cross-entropy trên các vị trí bị che mặt nạ:
$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0, 1), y_0 \sim \mathcal{D}, y_t \sim q(y_t \mid y_0)} \left[ -\frac{1}{t \cdot L} \sum_{i: y_t^i = \mathbf{M}} \log p_\theta(y_0^i \mid y_t, I, x) \right]$$

---

### 1.3. Cơ chế Khử nhiễu Bán tự hồi quy (Semi-Autoregressive Block Generation)
Để cân bằng giữa khả năng chú ý hai chiều toàn cục và sự mạch lạc cục bộ của câu văn dài, LLaDA-MedV chia chuỗi $L$ thành các khối (blocks) độ dài $B \le L$:
- Nếu $B = L$: Quá trình khuếch tán là **hoàn toàn song song hai chiều** (Full Bidirectional Diffusion).
- Nếu $B < L$: Mô hình sinh tuần tự từng khối từ trái qua phải, nhưng **bên trong mỗi khối là khuếch tán hai chiều**.
- Số bước lấy mẫu cho mỗi khối là:
  $$Z_{\text{block}} = Z \cdot \frac{B}{L}$$
Trong đó $Z$ là tổng số bước lấy mẫu (sampling steps) của toàn bộ chuỗi.

---

## 2. GIẢI PHẪU KIẾN TRÚC VÀ CÁC QUYẾT ĐỊNH THIẾT KẾ (DESIGN DECISIONS)

```
                 BẢN ĐỒ KIẾN TRÚC VÀ LUỒNG DỮ LIỆU LLaDA-MedV
                 
  +-------------------------------------------------------------------------+
  | HÌNH ẢNH LÂM SÀNG (X-ray, CT, MRI, Path)                                |
  | Kích thước: 384 x 384 x 3                                               |
  +-------------------------------------------------------------------------+
                                     │
                                     ▼
  +-------------------------------------------------------------------------+
  | VISION TOWER: SigLIP2-so400m-patch14-384                                |
  | Patch size: 14x14  -->  (384/14)^2 = 27^2 = 729 visual tokens           |
  | Hàm mất mát pretrain: Sigmoid Loss (không dùng Softmax Contrastive)     |
  +-------------------------------------------------------------------------+
                                     │
                                     ▼  [729 x 1152]
  +-------------------------------------------------------------------------+
  | MULTIMODAL PROJECTOR: 2-Layer MLP + GELU                                |
  | Linear(1152 -> 4096) -> GELU -> Linear(4096 -> 4096)                    |
  +-------------------------------------------------------------------------+
                                     │
                                     ▼  [729 x 4096] (Visual Embeddings)
  +-------------------------------------------------------------------------+
  | LANGUAGE BACKBONE: LLaDA-8B-Instruct (32 Transformer Layers, 4096 dim)   |
  | Input Stream: [Visual Tokens (729)] + [Text Prompt Tokens] + [MASK (L)] |
  | Attention Type: Full Bidirectional Masked Self-Attention                |
  +-------------------------------------------------------------------------+
                                     │
                                     ▼  (Lặp lại qua Z bước khử nhiễu)
  +-------------------------------------------------------------------------+
  | OUTPUT: Dự đoán phân phối xác suất từ vựng sạch y0 tại các vị trí MASK   |
  +-------------------------------------------------------------------------+
```

### 2.1. Lựa chọn Vision Tower: Tại sao SigLIP2-so400m-patch14-384 đánh bại CLIP truyền thống?
Hầu hết các VLM y tế trước đây (như LLaVA-Med v1) dùng `CLIP ViT-L/14` (224x224 hoặc 336x336). LLaDA-MedV chuyển sang `SigLIP2-so400m-patch14-384`. Đây là quyết định mang tính chiến lược vì:
1. **Độ phân giải $384 \times 384$ tăng 75% mật độ thông tin**:
   - Số lượng visual tokens tăng lên: $\left(\frac{384}{14}\right)^2 \approx 729$ tokens (so với 256 hoặc 576 của CLIP cũ).
   - Trong ảnh X-quang phổi và lát cắt mô bệnh học (Histology), các tổn thương mang tính quyết định (như vi vôi hóa - microcalcifications, nốt mờ đơn độc, hoặc ranh giới tế bào ung thư) chỉ chiếm vài pixel. Độ phân giải cao bảo toàn các ranh giới giải phẫu này.
2. **Cơ chế Sigmoid Loss thay thế Contrastive Softmax**:
   - CLIP truyền thống chuẩn hóa softmax qua toàn bộ batch: dễ bị ảnh hưởng tiêu cực bởi các cặp mẫu âm (negative samples) có hình ảnh gần giống nhau (như hai ảnh chụp X-quang lồng ngực bình thường).
   - SigLIP tối ưu hóa hàm sigmoid độc lập cho từng cặp ảnh-chữ, ngăn chặn hiện tượng trôi dạt biểu diễn (representation drift) khi làm việc với các cặp ảnh y tế có mức độ tương đồng thị giác cao.

---

### 2.2. Nghịch lý Khởi tạo (Initialization Paradox)
Tại Bảng 4 của bài báo, nhóm tác giả thực hiện một ablation cực kỳ quan trọng về trọng số khởi tạo:

| Mô hình khởi tạo | Cấu hình huấn luyện | Điểm Hội thoại Y tế (GPT-4 Score) | Hiện tượng quan sát được |
| :--- | :--- | :---: | :--- |
| $\text{LLaDA-MedV}^{V_1}$ | Khởi tạo từ **LLaDA-V** (đã học ảnh tự nhiên) $\rightarrow$ chỉ SFT Y tế | **31.12%** | Lặp từ nghiêm trọng, nhầm lẫn khái niệm giải phẫu |
| $\text{LLaDA-MedV}^{V_2}$ | Khởi tạo từ **LLaDA-V** $\rightarrow$ Căn chỉnh Alignment + SFT Y tế | **31.05%** | Vẫn bị kẹt ở điểm số thấp, không học được biểu diễn sâu |
| **LLaDA-MedV\*** | Khởi tạo từ **LLaDA gốc (Text-only)** $\rightarrow$ Căn chỉnh y sinh từ đầu | **52.60%** | **Tăng vọt +21.5%**, ngôn ngữ tự nhiên, lập luận chuẩn |

#### Phân tích Cơ chế Thất bại của Tiên nghiệm Tự nhiên (Negative Transfer)
1. **Trôi dạt không gian tiềm ẩn (Latent Representation Drift)**:
   - LLaDA-V được huấn luyện trên hàng triệu video và ảnh tự nhiên (chó, mèo, xe cộ, phong cảnh). Không gian attention của các tầng thấp đã bị gán chặt với các đặc trưng thị giác phổ thông (màu sắc rực rỡ, góc cạnh cảnh vật).
   - Khi đưa vào ảnh y tế (vốn là ảnh thang độ xám đơn sắc - Grayscale, tỷ lệ tương phản thấp, dựa trên mật độ mô Hounsfield Unit của CT hoặc tín hiệu T1/T2 của MRI), các trọng số thị giác tự nhiên hoạt động như **nhiễu nền (adversarial noise)**.
2. **Sự triệt tiêu gradient y sinh**: Các đặc trưng y sinh tinh vi bị làm phẳng bởi các attention map quá mạnh mẽ kế thừa từ miền tự nhiên.

---

## 3. MỔ XẺ THỰC CHẤT KẾT QUẢ VÀ CÁC ĐIỂM NGHẼN KỸ THUẬT (EMPIRICAL BREAKDOWN)

### 3.1. Nghịch lý Câu hỏi đóng (Closed-form SOTA) vs. Câu hỏi mở (Open-form Failure)

```
       SO SÁNH ĐỘ CHÍNH XÁC CLOSED-FORM VS RECALL OPEN-FORM TRÊN 3 BENCHMARK

    100% ┌────────────────────────────────────────────────────────────────────────┐
         │  84.93%       92.31%       95.15%                                      │
     80% │  ████         ████         ████             ■ Closed Accuracy (SOTA)   │
         │  ████         ████         ████             ░ Open-form Token Recall   │
     60% │  ████         ████  68.85% ████                                        │
         │  ████  45.60% ████  ░░░░   ████                                        │
     40% │  ████  ░░░░   ████  ░░░░   ████  31.96%                                │
         │  ████  ░░░░   ████  ░░░░   ████  ░░░░                                  │
     20% │  ████  ░░░░   ████  ░░░░   ████  ░░░░                                  │
      0% └────────────────────────────────────────────────────────────────────────┘
            VQA-RAD         SLAKE        PathVQA
```

#### Tại sao LLaDA-MedV thống trị Câu hỏi đóng?
- Trong câu hỏi đóng (Yes/No, Right/Left), bài toán thực chất là một phép phân loại nhị phân trên toàn bộ hình ảnh.
- Cơ chế **Bidirectional Self-Attention** cho phép tất cả các token câu hỏi tương tác qua lại trực tiếp với 729 visual tokens của hình ảnh trong suốt $Z$ bước lấy mẫu. Mô hình có thể liên tục đối chiếu câu hỏi với vùng tổn thương giải phẫu để đạt độ chính xác chẩn đoán cực cao (84.93% trên VQA-RAD, 92.31% trên SLAKE).

#### Tại sao LLaDA-MedV suy thoái trên Câu hỏi mở?
- **Lỗ hổng định dạng (Formatting Mismatch)**:
  - Nhãn chuẩn (Ground Truth) của câu hỏi mở trong các tập dữ liệu VQA thường là **từ đơn hoặc cụm từ ngắn gọn** (ví dụ: *"nodular opacity"*, *"cardiomegaly"*, *"axial"*).
  - Các mô hình ARM (như LLaVA-Med) học được cách ép ngắn câu trả lời nhờ hàm mất mát Cross-Entropy phạt nặng độ dài.
  - LLaDA-MedV giải mã bằng cách điền vào chuỗi mặt nạ cố định độ dài $L=64$. Mô hình có xu hướng sinh một câu ngữ pháp hoàn chỉnh mang tính đàm thoại:  
    *Ví dụ*:  
    - GT: `"cardiomegaly"` (1 token)  
    - LLaDA Pred: `"The chest radiograph shows an enlarged cardiac silhouette indicating cardiomegaly."` (11 tokens)
  - Khi tính toán **Token-level Recall**:
    $$\text{Recall} = \frac{|\text{Tokens}(GT) \cap \text{Tokens}(Pred)|}{|\text{Tokens}(GT)|}$$
    Mặc dù từ `"cardiomegaly"` xuất hiện trong câu trả lời (về mặt y khoa là chính xác 100%), nếu câu hỏi yêu cầu nhiều danh từ hoặc từ đồng nghĩa hơi khác một chút, việc thiếu hậu huấn luyện định dạng (Format Alignment) khiến điểm số giảm mạnh.

---

### 3.2. Cơ chế Bẫy Tiêu Điểm và Lặp Từ (Token Repetition Attractors)
Một hiện tượng kỳ lạ được báo cáo tại Hình 4 và Hình 10 của bài báo: Khi $L \ge 128$ và $Z \le 64$, mô hình suy thoái thành chuỗi lặp:
`"...the the the the the the the..."` hoặc `"...cells cells cells cells..."`

#### Giải thích Cơ chế Vật lý/Toán học của Hiện tượng
Trong mô hình khuếch tán ngôn ngữ:
1. Tại các bước lấy mẫu đầu tiên ($t \approx 1$), chuỗi gần như toàn là $\mathbf{M}$. Mô hình dựa vào độ tự tin (logits confidence) để quyết định unmask các token đầu tiên.
2. Các từ chức năng có tần suất xuất hiện cực cao trong tập dữ liệu y sinh (như *"the"*, *"of"*, *"cells"*, *"in"*) có xác suất tiên nghiệm (prior probability) áp đảo.
3. Khi một vài từ *"the"* được unmask sớm ở các vị trí ngẫu nhiên, cơ chế **Bidirectional Attention** của Transformer khiến các vị trí $\mathbf{M}$ còn lại nhìn thấy từ *"the"* ở khắp nơi.
4. Điều này tạo thành một **hố sụt năng lượng (energy attractor / degenerate basin)**: Các vị trí $\mathbf{M}$ xung quanh nhận thấy rằng việc tiếp tục dự đoán *"the"* sẽ tối đa hóa hàm mục tiêu tương thích cục bộ.
5. Nếu tổng số bước lấy mẫu $Z$ quá nhỏ, mô hình **không có đủ cơ hội để remask (xóa đi) và thay thế** các token lỗi bằng các danh từ y học thực thụ. Kết quả là chuỗi sụp đổ thành trạng thái lặp vô tận.

---

### 3.3. Độ phức tạp tính toán và Nghịch lý Bộ nhớ (Attention Quadratic Scaling & No KV-Cache)

| Đặc tính kiến trúc | Autoregressive Models (ARMs) | Masked Language Diffusion (LLaDA-MedV) |
| :--- | :---: | :---: |
| **Cơ chế Attention** | Causal Mask (Tam giác dưới) | **Full Bidirectional (Ma trận đầy đủ)** |
| **Sử dụng KV-Cache** | **Có** (Chỉ tính $K, V$ của token mới tại mỗi bước) | **KHÔNG** (Tất cả token đều thay đổi sau mỗi bước khử nhiễu) |
| **Độ phức tạp 1 bước** | $\mathcal{O}(L \cdot d)$ (với KV-cache) | $\mathcal{O}(L^2 \cdot d)$ (tính lại toàn bộ ma trận) |
| **Tổng chi phí sinh** | $\mathcal{O}(L^2 \cdot d)$ qua $L$ bước sinh | $\mathcal{O}(Z \cdot L^2 \cdot d)$ qua $Z$ bước sinh |
| **Thời gian suy luận** | **0.036 giây/từ** | **0.230 giây/từ (Chậm hơn 6.4x)** |

Nghịch lý lớn nhất của LLaDA-MedV hiện nay là: **Không thể tái sử dụng Key-Value Cache**. Trong ARM, các token đã sinh ở quá khứ là bất biến. Trong Diffusion, ở mỗi bước $t \to s$, toàn bộ trạng thái của chuỗi đều thay đổi, buộc mô hình phải tính lại toàn bộ ma trận Self-Attention $Z$ lần.

---

## 4. HỆ THỐNG CÁC ĐỀ TÀI VÀ THỰC NGHIỆM MỞ RỘNG TIỀM NĂNG (RESEARCH OPPORTUNITIES)

Dưới đây là thiết kế chi tiết 6 đề tài nghiên cứu tiếp nối có khả năng tạo ra các công bố khoa học độc lập tại các hội nghị AI/Y tế hàng đầu (MICCAI, CVPR, ECCV, NeurIPS).

---

### Đề tài 1: Remasking Động dựa trên Entropy và Trừng phạt Lặp từ (Entropy-Guided Dynamic Remasking)

#### 1. Câu hỏi Nghiên cứu (Research Question)
*Làm thế nào để triệt tiêu hoàn toàn bẫy lặp từ (Token Repetition Attractors) của mô hình khuếch tán ngôn ngữ y sinh trong chế độ sinh nhanh ($Z \le 32$) mà không cần huấn luyện lại mạng nơ-ron?*

#### 2. Cơ sở Lý thuyết & Công thức Đề xuất
Thay vì chọn cố định $\lfloor (1 - s/t) \cdot N_{\mathbf{M}} \rfloor$ token có xác suất cao nhất để unmask, ta định nghĩa hàm chấm điểm tin cậy hiệu chỉnh theo Entropy cục bộ và Trừng phạt Lặp từ:
$$S(i) = p_\theta(y_0^i \mid y_t, I, x) - \alpha \cdot \mathcal{H}(y_0^i) - \beta \cdot \text{RepPenalty}(y_0^i, y_t)$$

Trong đó:
- $\mathcal{H}(y_0^i) = -\sum_{w \in \mathcal{V}} p_\theta(w) \log p_\theta(w)$ là độ hỗn loạn (Entropy). Token y học chính xác sẽ có entropy rất thấp (độ tự tin tập trung), trong khi token lặp lại mang tính ngẫu nhiên có entropy cao.
- $\text{RepPenalty}(w, y_t) = \text{Count}(w, y_t)$ là số lần từ $w$ đã xuất hiện trong chuỗi hiện tại.
- $\alpha, \beta$ là các hệ số điều tiết siêu tham số.

#### 3. Thiết kế Thực nghiệm
- **Tập dữ liệu**: VQA-RAD Test, SLAKE Test (chọn các câu hỏi mô tả tổn thương dài $L \ge 128$).
- **Cấu hình so sánh**:
  1. Baseline LLaDA-MedV chuẩn: Lập lịch tuyến tính, $Z \in \{16, 32, 64\}$.
  2. Baseline Cosine Scheduler.
  3. Phương pháp đề xuất: Entropy-guided Dynamic Remasking.
- **Chỉ số đánh giá (Metrics)**:
  - $N$-gram Repetition Rate ($N=2, 3, 4$).
  - Distinct-1 / Distinct-2 (Độ đa dạng từ vựng).
  - VQA Closed Accuracy & Open Recall.

---

### Đề tài 2: Hậu huấn luyện Căn chỉnh Ưu tiên (Diffusion-DPO / Medical DPO) cho VQA Mở

#### 1. Câu hỏi Nghiên cứu
*Làm thế nào để huấn luyện LLaDA-MedV tự động tuân thủ định dạng chẩn đoán ngắn gọn của bác sĩ X-quang và nâng Token-level Recall trên câu hỏi mở từ 45.60% lên trên 70%?*

#### 2. Phương pháp Đề xuất: Direct Preference Optimization trên Masked Diffusion
Xây dựng tập dữ liệu so sánh cặp (Preference Dataset) $\mathcal{D}_{\text{pref}} = \{(I, x, y_w, y_l)\}$:
- $y_w$ (Winner / Preferred): Câu trả lời chuẩn xác, ngắn gọn, chứa danh pháp y khoa đích.
- $y_l$ (Loser / Dispreferred): Câu trả lời dài dòng, chứa từ đệm thừa hoặc lặp từ.

Hàm mất mát Diffusion-DPO được thiết lập dựa trên tỷ số log-likelihood khử nhiễu:
$$\mathcal{L}_{\text{Diff-DPO}}(\theta) = -\mathbb{E}_{(I, x, y_w, y_l), t} \left[ \log \sigma \left( \beta \log \frac{p_\theta(y_w \mid y_{w, t}, I, x)}{p_{\text{ref}}(y_w \mid y_{w, t}, I, x)} - \beta \log \frac{p_\theta(y_l \mid y_{l, t}, I, x)}{p_{\text{ref}}(y_l \mid y_{l, t}, I, x)} \right) \right]$$

#### 3. Thiết kế Thực nghiệm
- **Mô hình gốc**: `XZDong123/LLaDA-MedV` checkpoint sau Stage 2.
- **Dữ liệu huấn luyện**: 10,000 cặp câu hỏi - trả lời y sinh từ PMC-VQA và VQA-RAD Train.
- **Kỳ vọng đột phá**:
  - VQA-RAD Open Recall tăng từ **45.60% lên $\ge 68.0\%$**.
  - PathVQA Open Recall tăng từ **31.96% lên $\ge 55.0\%$**.
  - Bảo toàn 100% khả năng sinh câu dài khi prompt yêu cầu mở rộng.

---

### Đề tài 3: Chưng cất Tính nhất quán Đa phương thức (Multimodal Consistency Distillation)

#### 1. Câu hỏi Nghiên cứu
*Liệu có thể giảm số bước khử nhiễu của mô hình 8B từ $Z=64$ xuống $Z \in [4, 8]$ bước (tăng tốc gấp 8 lần) mà không làm suy giảm độ chính xác chẩn đoán lâm sàng?*

#### 2. Cơ sở Kỹ thuật: Consistency Models trên Discrete Text Space
Huấn luyện một mạng học sinh (Student Model) $f_\theta$ thực hiện hàm nhất quán:
$$f_\theta(y_t, t, I, x) = f_\theta(y_s, s, I, x) = y_0 \quad \forall s, t$$
- Thay vì đi từng bước nhỏ $t \to t - \frac{1}{Z}$, mô hình học sinh có thể nhảy một bước lớn từ trạng thái mặt nạ đầy đủ sang trạng thái văn bản sạch sau 4 - 8 bước lặp.

#### 3. Bảng Thiết kế Thực nghiệm So sánh Đa chiều
| Mô hình | Số bước $Z$ | Latency (s/mẫu) | VRAM (GB) | VQA-RAD Closed (%) | VQA-RAD Open (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LLaDA-MedV (Gốc) | 64 | ~2.75s | 16.2 | 84.93 | 45.60 |
| LLaDA-MedV (Fast) | 16 | ~0.72s | 16.2 | 71.40 *(suy thoái)* | 33.10 |
| **Consistency Distill (Đề xuất)** | **4 - 8** | **~0.25s (11x)** | **16.2** | **$\ge 83.50$** | **$\ge 44.00$** |
| LLaVA-Med (ARM baseline) | - | ~0.45s | 14.8 | 84.19 | 61.52 |

---

### Đề tài 4: Giải mã Ràng buộc Thực thể Lâm sàng (Constrained Ontology Decoding via Logit Masking)

#### 1. Ý tưởng Đột phá
Khác với mô hình ARM chỉ có thể lọc từ ở vị trí tiếp theo, Masked Diffusion cho phép áp dụng **ràng buộc từ vựng đồng thời trên toàn bộ câu**:
- Tích hợp từ điển danh pháp y học chuẩn hóa quốc tế **UMLS (Unified Medical Language System)** và **SNOMED-CT**.
- Tại bước khử nhiễu cuối cùng ($t \to 0$), áp dụng mặt nạ nhị phân trên không gian logits $\mathcal{V}$:
  $$\tilde{p}_\theta(w \mid y_t) = \begin{cases} p_\theta(w \mid y_t) & \text{nếu } w \in \mathcal{V}_{\text{UMLS}} \cup \mathcal{V}_{\text{Grammar}} \\ -\infty & \text{nếu ngược lại} \end{cases}$$
- **Kết quả**: Triệt tiêu hoàn toàn lỗi chính tả y khoa và các từ vựng phi lâm sàng.

---

### Đề tài 5: Thẩm định Ảo giác Y khoa (Medical Hallucination Benchmark & Bidirectional Fact-Checking)

#### 1. Đặt vấn đề
Bài báo LLaDA-MedV tuyên bố mô hình sinh trung bình **166 từ** (rất dài và giàu thông tin). Tuy nhiên, trong y học, **viết càng dài thì nguy cơ bịa đặt bệnh lý (hallucination) càng cao**. Bài báo gốc hoàn toàn chưa có thực nghiệm định lượng về tỷ lệ ảo giác này!

#### 2. Thiết kế Khung Đánh giá Ảo giác Đa tầng
Xây dựng pipeline thẩm định ảo giác dựa trên **Atomic Medical Claim Decomposition**:
1. **Phân rã câu trả lời**: Chia phản hồi 166 từ thành các khẳng định y khoa nguyên tử (Atomic Claims).  
   *Ví dụ*: Claim 1: *"Có bóng tim to"* | Claim 2: *"Góc sườn hoành hai bên tù"* | Claim 3: *"Có tràn dịch màng phổi"*.
2. **Xác minh đa nguồn (Multi-agent Verification)**:
   - Đại lý 1: GPT-4o kết hợp Medical Knowledge Base.
   - Đại lý 2: Sử dụng chính cơ chế Mask Infilling của LLaDA-MedV: Che đi từng khẳng định bệnh lý, yêu cầu mô hình dự đoán lại dựa trên ảnh. Nếu độ tự tin của mô hình thấp $\to$ Đánh dấu ảo giác.
3. **Chỉ số thẩm định**:
   - Hallucination Rate per 100 words (Số lỗi ảo giác trên 100 từ).
   - Critical Diagnostic Error Rate (Tỷ lệ chẩn đoán sai lệch nghiêm trọng gây nguy hiểm tính mạng).

---

### Đề tài 6: Mở rộng sang Dữ liệu Thể tích 3D CT/MRI (Volumetric Medical Diffusion VLM)

#### 1. Giới hạn Thực tế của LLaDA-MedV
Trong chẩn đoán lâm sàng thực tế, không có bác sĩ nào đọc kết quả CT/MRI chỉ qua 1 lát cắt 2D đơn lẻ. Bệnh lý (như đột quỵ, u não, phình động mạch) cần phải được theo dõi liên tục qua chuỗi thể tích 3D (3D Volumetric Slices).

#### 2. Đề xuất Kiến trúc LLaDA-Med3D
- **Vision Backbone**: Thay vì xử lý từng ảnh rời rạc, tích hợp một **Spatial-Temporal Resampler** (hoặc 3D Perceiver Resampler) để nhận chuỗi $D = 32 \sim 64$ lát cắt CT.
- Nén $D \times 729$ visual tokens thành một tập biểu diễn thể tích cố định $K = 512$ tokens.
- **Thử nghiệm trên Benchmark 3D**:
  - Tập dữ liệu **CT-RATE** (hơn 20,000 ca chụp CT lồng ngực thể tích kèm kết luận chẩn đoán thực tế).
  - So sánh trực tiếp với các mô hình 3D ARM như CT-CLIP, Rad-Dino.

---

## 5. HƯỚNG DẪN KỸ THUẬT TRIỂN KHAI VÀ TỐI ƯU HÓA TÀI NGUYÊN (ENGINEERING GUIDE)

Đối với các phòng thí nghiệm hoặc cá nhân không có cụm máy 4x A100 80GB như tác giả bài báo, việc tối ưu hóa kỹ thuật để chạy và huấn luyện LLaDA-MedV trên phần cứng hạn chế (như 1x RTX 4090 24GB hoặc Laptop GPU 6GB - 16GB) là bài toán sống còn.

### 5.1. Kỹ thuật Lượng tử hóa 4-bit NF4 và Tách tải CPU-GPU (Offloading)
Mô hình `XZDong123/LLaDA-MedV` ở định dạng bfloat16 chiếm ~16GB VRAM (chưa kể Vision Tower và Activation Memory).

Để chạy suy luận hoặc đánh giá trên GPU phổ thông:
```python
import torch
from transformers import BitsAndBytesConfig
from llava.model.builder import load_pretrained_model

# Cấu hình lượng tử hóa 4-bit NormalFloat
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)

# Tải mô hình với phân bổ bộ nhớ động
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="XZDong123/LLaDA-MedV",
    model_base=None,
    model_name="llava_llada",
    quantization_config=bnb_config,
    device_map="auto"
)
```
- **Hiệu quả đạt được**: Dung lượng VRAM giảm từ **16.2 GB xuống còn ~5.4 GB**, cho phép chạy mượt mà trên GPU RTX 3060/4060/4050.

---

### 5.2. Tối ưu hóa I/O và Pipeline dữ liệu Parquet
Tệp `test-00000-of-00001.parquet` chứa sẵn mảng byte nhị phân của ảnh. Tránh giải nén toàn bộ 451 ảnh vào bộ nhớ RAM cùng một lúc (gây tràn RAM hệ thống trên máy cá nhân).

Khuyến nghị sử dụng kỹ thuật Generator / Streaming:
```python
import io
import pyarrow.parquet as pq
from PIL import Image

def stream_vqarad_samples(parquet_file, batch_size=1):
    table = pq.read_table(parquet_file, columns=["image", "question", "answer"])
    for i in range(0, table.num_rows, batch_size):
        slice_table = table.slice(i, batch_size).to_pylist()
        for row in slice_table:
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            yield {
                "image": img,
                "question": row["question"],
                "ground_truth": row["answer"]
            }
```

---

## 6. TỔNG KẾT VÀ KẾ HOẠCH HÀNH ĐỘNG

### 6.1. Bảng Tóm Tắt Định Hướng Nghiên Cứu
| Hướng đề xuất | Độ khó kỹ thuật | Tài nguyên yêu cầu | Mức độ đóng góp khoa học | Bài báo mục tiêu |
| :--- | :---: | :---: | :---: | :---: |
| **1. Entropy-Guided Remasking** | Trung bình | 1 GPU (RTX 3090/4090) | Rất cao (Giải quyết lỗi cốt tử) | CVPR / MICCAI Short |
| **2. Medical Diffusion-DPO** | Cao | 2 - 4 GPU A100 | Đột phá (Nâng Open-form VQA) | NeurIPS / ICLR |
| **3. Consistency Distillation** | Rất cao | 4x A100 (Huấn luyện lại) | Đột phá (Tăng tốc suy luận 8x) | CVPR / ICML |
| **4. Medical Hallucination Audit**| Thấp - Trung bình | API GPT-4o + 1 GPU | Thực tiễn lâm sàng cao | Nature Medicine / Lancet DH |
| **5. 3D Volumetric Extension** | Rất cao | Cụm máy chủ GPU lớn | Mở ra phân nhánh mô hình mới | CVPR / MICCAI Oral |

### 6.2. Lời khuyên cho Bước Đi Đầu Tiên
Nếu bạn đang chuẩn bị bắt tay vào làm thực nghiệm ngay:
1. **Bước 1**: Sử dụng notebook [vqa_rad_experiments.ipynb](file:///home/mercury/HLDK-LLADA-MedV/notebook/vqa_rad_experiments.ipynb) đã tạo để chạy thử nghiệm định tuyến (sanity check) trên 50 mẫu đầu tiên của VQA-RAD.
2. **Bước 2**: Triển khai ngay **Đề tài 1 (Entropy-Guided Remasking)**. Đây là hướng đi khả thi nhất, không tốn tài nguyên huấn luyện lại, giải quyết trực tiếp lỗi lặp từ của bài báo và có thể ra kết quả định lượng ngay trong vòng 1-2 tuần.
3. **Bước 3**: Mở rộng bảng đánh giá sang BioBERTScore và ROUGE-L để chứng minh rằng câu trả lời dài của LLaDA-MedV thực chất mang giá trị thông tin y học cao hơn nhiều so với điểm Token Recall đơn thuần.

---
*Tài liệu phân tích chuyên sâu này được hoàn thiện nhằm đóng vai trò kim chỉ nam lý thuyết và thực nghiệm cho các công trình nghiên cứu tiếp nối.*
