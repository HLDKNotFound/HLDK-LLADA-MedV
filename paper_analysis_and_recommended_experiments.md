# 📑 BÁO CÁO PHÂN TÍCH CHUYÊN SÂU BÀI BÁO LLaDA-MedV VÀ ĐỀ XUẤT HỆ THỐNG THỰC NGHIỆM TIẾP THEO

> **Tên bài báo**: *LLaDA-MedV: Exploring Large Language Diffusion Models for Biomedical Image Understanding*  
> **Tác giả**: Xuanzhao Dong, Wenhui Zhu, Xiwen Chen, Zhipeng Wang, Peijie Qiu, Shao Tang, Xin Li, Yalin Wang  
> **Kỷ yếu**: **CVPR 2026** (arXiv: [2508.01617](https://arxiv.org/abs/2508.01617))  
> **Mã nguồn & Trọng số**: [GitHub](https://github.com/LLM-VLM-GSL/LLaDA-MedV) | [HuggingFace](https://huggingface.co/XZDong123/LLaDA-MedV)

---

## MỤC LỤC
1. [TỔNG QUAN VÀ ĐỘNG LỰC NGHIÊN CỨU](#1-tổng-quan-và-động-lực-nghiên-cứu)
2. [KIẾN TRÚC MÔ HÌNH VÀ QUY TRÌNH HUẤN LUYỆN 3 GIAI ĐOẠN](#2-kiến-trúc-mô-hình-và-quy-trình-huấn-luyện-3-giai-đoạn)
3. [PHÂN TÍCH KẾT QUẢ ĐẠT ĐƯỢC VÀ CÁC PHÁT HIỆN CHÍNH](#3-phân-tích-kết-quả-đạt-được-và-các-phát-hiện-chính)
4. [PHÂN TÍCH HẠN CHẾ VÀ ĐIỂM YẾU CỦA BÀI BÁO (CRITICAL LIMITATIONS)](#4-phân-tích-hạn-chế-và-điểm-yếu-của-bài-báo-critical-limitations)
5. [ĐỀ XUẤT TOÀN DIỆN CÁC THỰC NGHIỆM NÊN BỔ SUNG (RECOMMENDED EXPERIMENTS)](#5-đề-xuất-toàn-diện-các-thực-nghiệm-nên-bổ-sung-recommended-experiments)
   - [Nhóm A: Khắc phục hiện tượng lặp từ và tối ưu cơ chế lấy mẫu Diffusion](#nhóm-a-khắc-phục-hiện-tượng-lặp-từ-và-tối-ưu-cơ-chế-lấy-mẫu-diffusion)
   - [Nhóm B: Nâng cao hiệu năng câu hỏi mở (Open-form VQA) & Post-Training Alignment](#nhóm-b-nâng-cao-hiệu-năng-câu-hỏi-mở-open-form-vqa--post-training-alignment)
   - [Nhóm C: Đánh giá độ tin cậy y khoa và ảo giác lâm sàng (Hallucination & Factuality)](#nhóm-c-đánh-giá-độ-tin-cậy-y-khoa-và-ảo-giác-lâm-sàng-hallucination--factuality)
   - [Nhóm D: Tối ưu hóa hiệu năng tính toán và suy luận thời gian thực](#nhóm-d-tối-ưu-hóa-hiệu-năng-tính-toán-và-suy-luận-thời-gian-thực)
   - [Nhóm E: Mở rộng sang ảnh thể tích 3D (CT/MRI Multi-slice) và chuỗi đa ảnh](#nhóm-e-mở-rộng-sang-ảnh-thể-tích-3d-ctmri-multi-slice-và-chuỗi-đa-ảnh)
6. [KẾT LUẬN VÀ LỘ TRÌNH TRIỂN KHAI](#6-kết-luận-và-lộ-trình-triển-khai)

---

## 1. TỔNG QUAN VÀ ĐỘNG LỰC NGHIÊN CỨU

### 1.1. Bối cảnh bài toán
Trong những năm gần đây, các mô hình thị giác - ngôn ngữ y sinh (Biomedical Vision-Language Models - VLMs) như **LLaVA-Med**, **Med-Flamingo**, **BiomedCLIP**, **MedVLM-R1** đều được xây dựng dựa trên mô hình tự hồi quy (**Autoregressive Models - ARMs**). 

Mặc dù ARMs rất thành công, trong ngữ cảnh lâm sàng thực tế, chúng bộc lộ **2 nhược điểm cố hữu nghiêm trọng**:
1. **Khả năng kiểm soát độ dài phản hồi kém (Lack of Length Controllability)**:
   - Các mô hình ARM thường dựa vào heuristic như giới hạn độ dài `max_tokens` hoặc prompt hệ thống (*"Hãy trả lời chi tiết ít nhất 200 từ"*).
   - Tuy nhiên, khi huấn luyện trên dữ liệu VQA ngắn gọn, ARM có xu hướng dự đoán token kết thúc chuỗi `<eos>` quá sớm. Dù prompt yêu cầu dài, câu trả lời thực tế chỉ đạt ~36 - 40 từ, dẫn đến thiếu hụt bối cảnh y khoa, thiếu giải thích về cơ chế bệnh sinh, khuyến nghị chẩn đoán phân biệt và tư vấn an toàn.
2. **Cơ chế sinh đơn hướng trái-qua-phải (Left-to-right Unidirectional Generation)**:
   - ARMs chỉ nhìn về các token quá khứ, không có khả năng quan sát toàn cục hay tinh chỉnh ngược (global bidirectional refinement). Một sai sót từ vựng y khoa ở đầu câu sẽ kéo theo chuỗi chẩn đoán sai lệch ở các bước tiếp theo.

### 1.2. Đột phá từ Mô hình Khuếch tán Ngôn ngữ (Large Language Diffusion Models)
**LLaDA-MedV** là công trình tiên phong đầu tiên đưa mô hình khuếch tán ngôn ngữ mặt nạ (**Masked Discrete Diffusion**) vào bài toán hiểu ảnh y sinh:
- **Nguyên lý hoạt động**: Quá trình sinh bắt đầu từ một chuỗi token hoàn toàn bị che kín mặt nạ (fully masked sequence $\mathbf{M}$) có độ dài xác định trước $L$. Mô hình dần dần khử nhiễu (denoise) qua $Z$ bước lấy mẫu để điền các từ có độ tin cậy cao vào chuỗi.
- **Lợi thế vượt trội**:
  - **Kiểm soát độ dài chính xác tuyệt đối**: Cho phép sinh câu trả lời dài trung bình **166 từ** (gấp 4.6 lần so với 36 từ của LLaVA-Med), cung cấp đầy đủ chi tiết giải phẫu học, cơ sở chẩn đoán và khuyến nghị lâm sàng.
  - **Ngữ cảnh hai chiều (Bidirectional Attention)**: Mọi vị trí token đều được quan sát toàn bộ ngữ cảnh xung quanh, cho phép mô hình kiểm chứng chéo thông tin.

---

## 2. KIẾN TRÚC MÔ HÌNH VÀ QUY TRÌNH HUẤN LUYỆN 3 GIAI ĐOẠN

```
+-----------------------------------------------------------------------------------+
|                              KIẾN TRÚC LLaDA-MedV                                 |
+-----------------------------------------------------------------------------------+
|  [Ảnh Y Khoa (X-quang, CT, MRI, Mô học)]                                          |
|        │                                                                          |
|        ▼                                                                          |
|  [Vision Tower: SigLIP2-so400m-patch14-384] (384x384, trích xuất đặc trưng sắc nét)|
|        │                                                                          |
|        ▼                                                                          |
|  [Projector: 2-layer MLP + GELU] (Ánh xạ không gian ảnh sang không gian văn bản)  |
|        │                                                                          |
|        ▼ (Visual Tokens)                                                          |
|  [Language Backbone: LLaDA-8B-Instruct (Masked Diffusion Transformer)]            |
|        ▲                                                                          |
|        │ (Prompt & Noisy Sequence xt)                                             |
|  [Text Input: "<image>\n" + Câu hỏi lâm sàng] + [Chuỗi mặt nạ độ dài L]           |
+-----------------------------------------------------------------------------------+
```

### 2.1. Cấu trúc Mô-đun
- **Vision Tower**: `google/siglip2-so400m-patch14-384`. Khác với CLIP ViT-L/14 truyền thống (độ phân giải 224x224 hoặc 336x336), SigLIP2 xử lý ảnh ở độ phân giải $384 \times 384$, giúp nhận diện rõ ràng các tổn thương vi mô trong ảnh giải phẫu bệnh (Histology) và nốt mờ nhỏ trong X-quang phổi.
- **Multimodal Projector**: MLP 2 tầng với hàm kích hoạt GELU, gọn nhẹ nhưng đủ khả năng chuyển đổi không gian đặc trưng thị giác sang không gian embedding của ngôn ngữ.
- **Language Backbone**: `LLaDA-8B-Instruct` với cơ chế Semi-Autoregressive Generation (chia chuỗi $L$ thành các block $B$).

### 2.2. Chiến lược Huấn luyện 3 giai đoạn (Training Pipeline)
1. **Giai đoạn 1 - Đồng bộ hóa đa phương thức (Modality Alignment)**:
   - Dữ liệu: 600,000 cặp ảnh - văn bản y sinh lọc từ PubMed Central (PMC).
   - Mục tiêu: Chỉ huấn luyện Projector để căn chỉnh đặc trưng hình ảnh của SigLIP2 vào không gian từ vựng của LLaDA, đóng băng Vision Tower và Language Backbone.
2. **Giai đoạn 2 - Tinh chỉnh hướng dẫn thị giác y sinh (Visual Instruction Tuning / SFT)**:
   - Dữ liệu: 60,000 mẫu hội thoại đa lượt có trích dẫn nhúng (inline-mention dialogue dataset).
   - Mục tiêu: Mở khóa Language Backbone và Projector, huấn luyện mô hình khả năng suy luận lâm sàng và hội thoại mở.
3. **Giai đoạn 3 - Tinh chỉnh chuyên sâu từng tác vụ (Task-specific Fine-Tuning - SD-SFT)**:
   - Dữ liệu: Tập huấn luyện chính thức của 3 benchmark hạ nguồn: **VQA-RAD** (1,797 QA), **SLAKE** (4,919 QA), **PathVQA** (19,755 QA).
   - Định dạng: Chuyển đổi mỗi mẫu VQA thành hội thoại đơn lượt (Single-turn QA).

---

## 3. PHÂN TÍCH KẾT QUẢ ĐẠT ĐƯỢC VÀ CÁC PHÁT HIỆN CHÍNH

### 3.1. Đạt kỷ lục State-of-the-Art (SOTA) trên các câu hỏi đóng (Closed-form VQA)
Trên tập con câu hỏi đóng (Yes/No, câu hỏi trắc nghiệm nhị phân), LLaDA-MedV thiết lập kỷ lục độ chính xác mới vượt qua tất cả các mô hình ARM hàng đầu:

| Benchmark | LLaVA | LLaVA-Med | BiomedCLIP | M2I2 | **LLaDA-MedV (Ours)** | Mức tăng so với LLaVA-Med |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VQA-RAD (Radiology)** | 65.07% | 84.19% | 79.80% | 83.50% | **84.93%** | **+0.74%** |
| **SLAKE (Knowledge-Enhanced)** | 63.22% | 85.34% | 89.70% | 91.10% | **92.31%** | **+6.97%** |
| **PathVQA (Pathology)** | 63.20% | 91.21% | - | 88.00% | **95.15%** | **+3.94%** |

### 3.2. Vượt trội trong hội thoại y tế mở (Open-ended Biomedical Visual Chatbot)
Đánh giá theo chuẩn GPT-4 Evaluation đối chiếu với câu trả lời chuyên gia:
- **Tổng thể**: LLaDA-MedV đạt điểm tương đối **52.605%**, vượt qua LLaVA-Med (**44.750%**, tương đương tăng **+7.855%** tương đối) và MedVLM-R1 (**50.154%**).
- **Đặc biệt ở các miền**:
  - X-quang ngực (CXR): **64.352%** (so với 40.819% của LLaVA-Med, tăng vọt +23.53%).
  - Chụp cắt lớp (CT): **50.268%** (so với 42.083% của LLaVA-Med).

### 3.3. Tầm quan trọng của Trọng số Khởi tạo (Initialization Weights Analysis)
Bài báo thực hiện một phát hiện quan trọng tại Bảng 4:
- Khi khởi tạo từ **LLaDA-V** (mô hình VLM diffusion đã được train trên ảnh tự nhiên thông thường), hiệu năng hội thoại y tế chỉ đạt **31.12% - 31.05%**.
- Khi khởi tạo từ **LLaDA gốc** và thực hiện căn chỉnh y sinh từ đầu (LLaDA-MedV*), điểm số đạt **52.605%**.
- *Kết luận*: Các tiên nghiệm (priors) của miền ảnh tự nhiên có thể gây xung đột hoặc cản trở ngữ nghĩa tinh tế của hình ảnh y sinh học.

---

## 4. PHÂN TÍCH HẠN CHẾ VÀ ĐIỂM YẾU CỦA BÀI BÁO (CRITICAL LIMITATIONS)

Mặc dù LLaDA-MedV là bước tiến lớn, nghiên cứu còn tồn tại **4 điểm yếu cốt tử** mà nhóm tác giả đã thừa nhận hoặc chưa giải quyết triệt để:

### 4.1. Hiệu năng câu hỏi mở (Open-form VQA) còn kém hơn ARMs
- **Hiện tượng**: Trong Bảng 2, điểm Token-level Recall của LLaDA-MedV trên câu hỏi mở thấp hơn LLaVA-Med và các mô hình chuyên biệt:
  - VQA-RAD Open: **45.60%** (so với 61.52% của LLaVA-Med, 79.19% của Q2ATransformer).
  - SLAKE Open: **68.85%** (so với 83.08% của LLaVA-Med).
  - PathVQA Open: **31.96%** (so với 37.95% của LLaVA-Med, 71.49% của VL Encoder-Decoder).
- **Nguyên nhân sâu xa**: 
  - LLaDA chưa có giai đoạn **Post-training (RLHF/DPO)** để học cách tuân thủ định dạng ngặt nghèo (format constraint).
  - Các baseline ARM thường mô hình hóa câu hỏi mở như một bài toán phân loại trên tập từ vựng ứng viên cố định (predefined answer set derived from training split). Ngược lại, LLaDA-MedV sinh từ vựng tự do, có xu hướng diễn giải thành một cụm từ dài hoặc câu hoàn chỉnh, dẫn đến điểm so khớp từ đơn bị phạt nặng.

### 4.2. Hiện tượng lặp từ nghiêm trọng khi bước lấy mẫu thấp (Token Repetition Issue)
- **Hiện tượng**: Khi sinh câu trả lời dài ($L \ge 128$) nhưng số bước lấy mẫu $Z$ nhỏ ($Z \le 64$), mô hình gặp hiện tượng lặp lại liên tục một token chức năng:
  *Ví dụ*: `"...the coronavirus coronavirus the the the the the the the the the..."` (Hình 10 & 4 trong bài báo).
- **Cơ chế lỗi**: Khi $Z$ nhỏ, mô hình không có đủ các bước unmasking/remasking lặp lại để phát hiện và sửa chữa các token lỗi có độ tự tin thấp ở giai đoạn đầu. Lỗi sớm này bị đóng băng và lan truyền ra toàn bộ chuỗi.

### 4.3. Chi phí tính toán và độ trễ suy luận cao (Inference Overhead)
- LLaDA-MedV cần $Z$ bước lan truyền thuận qua toàn bộ mạng 8B (với $Z=64$ hoặc $Z=256$).
- Thời gian sinh trung bình là **0.230 giây/từ**, chậm hơn **6.4 lần** so với ARM thông thường (**0.036 giây/từ**). Điều này gây cản trở lớn khi triển khai thực tế trên các máy tính trạm bệnh viện hoặc thiết bị biên.

### 4.4. Giới hạn đánh giá: Chỉ kiểm thử ảnh 2D tĩnh
- Tất cả các benchmark (VQA-RAD, SLAKE, PathVQA) đều là ảnh lát cắt 2D đơn lẻ.
- Trong thực tế chẩn đoán hình ảnh, bác sĩ X-quang chẩn đoán dựa trên **toàn bộ khối thể tích 3D (3D CT/MRI volumetric scans với hàng trăm lát cắt)** hoặc so sánh đối chiếu đa ảnh (prior vs current study). LLaDA-MedV chưa có cơ chế tiếp nhận 3D hoặc multi-image.

---

## 5. ĐỀ XUẤT TOÀN DIỆN CÁC THỰC NGHIỆM NÊN BỔ SUNG (RECOMMENDED EXPERIMENTS)

Dưới đây là thiết kế chi tiết các hệ thống thực nghiệm cần thiết để giải quyết triệt để các hạn chế trên, tối ưu hóa mô hình và mở rộng đóng góp khoa học:

```
+-----------------------------------------------------------------------------------+
|               HỆ THỐNG CÁC THỰC NGHIỆM MỞ RỘNG ĐƯỢC ĐỀ XUẤT                       |
+-----------------------------------------------------------------------------------+
|  [Nhóm A: Tối ưu Diffusion & Khắc phục Lặp từ]                                    |
|     ├── A1: Entropy-guided Dynamic Remasking Schedule                             |
|     ├── A2: Classifier-Free Guidance (CFG) cho Visual Grounding                   |
|     └── A3: Consistency Distillation (Giảm từ 64 bước xuống 8-16 bước)           |
+-----------------------------------------------------------------------------------+
|  [Nhóm B: Nâng cao Open-form VQA & Post-Training Alignment]                       |
|     ├── B1: Direct Preference Optimization (DPO / SimPO) cho ngắn gọn định dạng  |
|     ├── B2: Constrained Diffusion Decoding theo từ điển thực thể y tế (UMLS)      |
|     └── B3: Mở rộng bộ độ đo: BioBERTScore, ROUGE-L, Clinical BLEU               |
+-----------------------------------------------------------------------------------+
|  [Nhóm C: Đánh giá Tin cậy Lâm sàng & Ảo giác (Safety & Hallucination)]            |
|     ├── C1: Kiểm định Tỷ lệ Ảo giác Y khoa (Medical Hallucination Rate - MedFact)|
|     └── C2: Đánh giá Blind-test bởi Hội đồng Bác sĩ X-quang                       |
+-----------------------------------------------------------------------------------+
|  [Nhóm D: Lượng tử hóa và Tối ưu Biên (Edge Deployment & Quantization)]          |
|     ├── D1: Lượng tử hóa 4-bit AWQ / bitsandbytes NF4                             |
|     └── D2: Benchmark bộ nhớ VRAM và TPS (Tokens Per Second) trên RTX 4090/A100   |
+-----------------------------------------------------------------------------------+
|  [Nhóm E: Mở rộng Đa ảnh và Thể tích 3D (Multi-slice & 3D CT/MRI)]                |
|     └── E1: Kiến trúc Temporal/Inter-slice Attention trên chuỗi lát cắt CT/MRI   |
+-----------------------------------------------------------------------------------+
```

---

### Nhóm A: Khắc phục hiện tượng lặp từ và tối ưu cơ chế lấy mẫu Diffusion

#### Thực nghiệm A1: Lập lịch Remasking Động dựa trên Entropy (Entropy-guided Dynamic Remasking)
- **Mục tiêu**: Loại bỏ triệt để hiện tượng lặp từ ("the the the...") khi $Z$ thấp mà không làm tăng thời gian suy luận.
- **Phương pháp**:
  - Hiện tại, LLaDA-MedV dùng lịch unmasking tuyến tính hoặc cosine cố định.
  - Đề xuất: Tại mỗi bước lấy mẫu $t$, tính toán phân phối entropy dự đoán của các token. Những token nào bị trùng lặp trong $N$-gram lân cận hoặc có entropy quá cao (độ tin cậy thấp) sẽ bị đưa vào danh sách **bắt buộc remask** ở bước tiếp theo ($t-1$).
- **Độ đo đánh giá**:
  - Tỷ lệ lặp từ n-gram ($N$-gram Repetition Rate với $N \in \{2, 3, 4\}$).
  - Tỷ lệ xuất hiện từ vô nghĩa (Degeneration Rate).
  - Điểm VQA Accuracy giữ nguyên hoặc tăng.

#### Thực nghiệm A2: Tinh chỉnh Classifier-Free Guidance (CFG) cho Text Diffusion
- **Mục tiêu**: Tăng cường mức độ bám sát câu hỏi và đặc trưng hình ảnh bệnh lý, tránh tình trạng mô hình tự sinh văn bản chung chung.
- **Phương pháp**:
  - Điều chỉnh xác suất dropout điều kiện thị giác $p_{\text{uncond}} = 0.1$ trong lúc huấn luyện.
  - Khi suy luận, áp dụng công thức hướng dẫn:
    $$\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, \text{Image, Question}) - \epsilon_\theta(x_t, \emptyset))$$
  - Thử nghiệm quét dải trọng số hướng dẫn $s \in [1.0, 1.5, 2.0, 3.0]$.
- **Kỳ vọng**: Điểm Closed Accuracy trên VQA-RAD vượt mốc **86.0%**, cải thiện khả năng phát hiện các tổn thương nhỏ.

#### Thực nghiệm A3: Chưng cất tính nhất quán (Consistency Distillation) để tăng tốc suy luận 4x - 8x
- **Mục tiêu**: Giảm số bước $Z$ từ $64$ bước xuống còn **8 đến 16 bước**, đưa tốc độ suy luận về gần ngang với ARM ($< 0.05$ giây/từ).
- **Phương pháp**: Huấn luyện một Consistency Student Network học cách ánh xạ trực tiếp từ bất kỳ trạng thái $x_t$ về trạng thái sạch $x_0$ sau 1 bước nhảy lớn, dựa trên trọng số đã huấn luyện của LLaDA-MedV.
- **Độ đo đánh giá**: Latency (ms/token), VRAM tiêu thụ, độ chênh lệch điểm số VQA so với mô hình gốc 64 bước.

---

### Nhóm B: Nâng cao hiệu năng câu hỏi mở (Open-form VQA) & Post-Training Alignment

#### Thực nghiệm B1: Tinh chỉnh ưu đãi trực tiếp (DPO / SimPO) cho bài toán VQA Y tế
- **Mục tiêu**: Nâng điểm Open-form Token Recall từ **45.60% lên trên 65%**, giải quyết triệt để điểm yếu lớn nhất của bài báo.
- **Phương pháp**:
  - Xây dựng tập dữ liệu so sánh cặp (Pairwise Preference Dataset):
    - Câu trả lời được chọn ($y_w$): Ngắn gọn, chứa chính xác thuật ngữ y khoa đích (ví dụ: *"cardiomegaly"* hoặc *"left lower lobe pneumonia"*).
    - Câu trả lời bị từ chối ($y_l$): Dài dòng, lặp từ, hoặc vòng vo không chứa từ khóa chẩn đoán.
  - Áp dụng Direct Preference Optimization (DPO) được điều chỉnh cho Masked Diffusion Loss.
- **Độ đo đánh giá**: Token Recall trên VQA-RAD Open, SLAKE Open, PathVQA Open.

#### Thực nghiệm B2: Giải mã ràng buộc theo từ điển thực thể y khoa (UMLS-Constrained Decoding)
- **Mục tiêu**: Đảm bảo câu trả lời mở luôn chứa đúng thuật ngữ danh pháp y học quốc tế chuẩn (UMLS / SNOMED-CT).
- **Phương pháp**:
  - Khi khử nhiễu các token ở bước cuối cùng, áp dụng một bộ lọc mặt nạ logits (Logit Masking) chỉ cho phép giải mã các khái niệm y tế thuộc ontology lâm sàng phù hợp với câu hỏi.
- **Độ đo đánh giá**: Tỷ lệ câu trả lời khớp với thực thể UMLS (Concept Exact Match Rate).

#### Thực nghiệm B3: Đa dạng hóa bộ độ đo đánh giá câu hỏi mở
- **Mục tiêu**: Đánh giá thực chất chất lượng câu trả lời mở, tránh việc chỉ dùng Token Recall làm sai lệch giá trị của câu trả lời dài đầy đủ ngữ cảnh.
- **Phương pháp**:
  - Triển khai song song: **BioBERTScore**, **BLEU-1/2/3/4**, **ROUGE-L**, **METEOR**, và **GPT-4 Clinician Score**.
  - So sánh tương quan giữa các độ đo này với đánh giá của bác sĩ thực tế.

---

### Nhóm C: Đánh giá độ tin cậy y khoa và ảo giác lâm sàng (Hallucination & Factuality)

#### Thực nghiệm C1: Đánh giá Tỷ lệ Ảo giác Y khoa (Medical Hallucination Benchmark)
- **Mục tiêu**: Kiểm tra xem việc mô hình sinh câu trả lời dài (166 từ) có làm tăng tỷ lệ bịa đặt bệnh lý hay không.
- **Phương pháp**:
  - Sử dụng khung đánh giá **MedFact** hoặc mô hình GPT-4o với prompt chuyên gia thẩm định: Phân tích từng khẳng định y khoa trong câu trả lời (Atomic Claim Verification) thành 3 trạng thái:
    1. *Hỗ trợ bởi hình ảnh* (Supported by Image).
    2. *Mâu thuẫn với hình ảnh* (Contradicted by Image - Ảo giác nghiêm trọng).
    3. *Không thể xác định từ hình ảnh đơn lẻ* (Unverifiable).
- **Kỳ vọng**: Chứng minh rằng LLaDA-MedV có tỷ lệ Contradiction thấp hơn LLaVA-Med nhờ cơ chế chú ý hai chiều.

#### Thực nghiệm C2: Đánh giá mù đôi bởi chuyên gia X-quang (Double-Blind Clinical Study)
- **Phương pháp**: 
  - Chọn ngẫu nhiên 100 ca khó từ VQA-RAD và SLAKE.
  - Ẩn danh nguồn gốc câu trả lời giữa: (1) LLaVA-Med, (2) LLaDA-MedV, (3) Bác sĩ X-quang thực tế.
  - 3 Bác sĩ chẩn đoán hình ảnh độc lập chấm điểm theo thang 1 - 5 trên 4 tiêu chí: Tính chính xác (Accuracy), Độ chi tiết hữu ích (Clinical Utility), Tính an toàn (Safety), và Mức độ tin cậy (Completeness).

---

### Nhóm D: Tối ưu hóa hiệu năng tính toán và suy luận thời gian thực

#### Thực nghiệm D1: Lượng tử hóa 4-bit và 8-bit (NF4 / AWQ / GPTQ)
- **Mục tiêu**: Cho phép chạy LLaDA-MedV trên các GPU phổ thông hoặc laptop cá nhân có VRAM $\le 8\text{GB}$ (như RTX 4050/4060) mà người dùng nghiên cứu thông thường có thể tiếp cận được.
- **Phương pháp**:
  - Lượng tử hóa trọng số Language Backbone 8B sang **4-bit NormalFloat (NF4)** với BitsAndBytes.
  - Đánh giá sự suy giảm điểm số trên VQA-RAD Closed & Open.
- **Độ đo**: VRAM tiêu thụ thực tế (GB), tốc độ suy luận (samples/min), độ suy giảm điểm Accuracy (< 1.5% là đạt yêu cầu).

---

### Nhóm E: Mở rộng sang ảnh thể tích 3D (CT/MRI Multi-slice) và chuỗi đa ảnh

#### Thực nghiệm E1: Tích hợp Kiến trúc Chú ý Liên lát cắt (Inter-slice Spatial-Temporal Attention)
- **Mục tiêu**: Mở rộng khả năng của Large Language Diffusion Model từ ảnh 2D sang chuỗi CT/MRI 3D (32 - 64 lát cắt).
- **Phương pháp**:
  - Trích xuất đặc trưng từng lát cắt bằng SigLIP2, sau đó qua một module 3D Resampler / Temporal Perceiver để nén chuỗi thể tích thành số lượng visual tokens cố định ($K=256$).
  - Truyền các visual token này làm điều kiện cho LLaDA Diffusion Backbone.
- **Tập dữ liệu đề xuất**: **CT-RATE** (Chest CT Volumes with Radiology Reports) hoặc **COV-CTR**.

---

## 6. KẾT LUẬN VÀ LỘ TRÌNH TRIỂN KHAI

### 6.1. Đánh giá tổng kết bài báo
| Tiêu chí | Đánh giá | Chi tiết |
| :--- | :---: | :--- |
| **Tính mới khoa học (Novelty)** | ⭐⭐⭐⭐⭐ | Công trình đầu tiên mở đường cho Large Language Diffusion trong miền thị giác y sinh. |
| **Hiệu năng câu hỏi đóng (Closed VQA)** | ⭐⭐⭐⭐⭐ | Thiết lập kỷ lục SOTA ấn tượng trên VQA-RAD (84.93%), SLAKE (92.31%), PathVQA (95.15%). |
| **Chất lượng hội thoại mở (Conversation)** | ⭐⭐⭐⭐☆ | Câu trả lời dài, phong phú ngữ cảnh lâm sàng gấp 4.6 lần so với LLaVA-Med. |
| **Hiệu năng câu hỏi mở (Open-form VQA)** | ⭐⭐⭐☆☆ | Điểm số còn thấp do thiếu post-training và thiếu ràng buộc từ điển ứng viên. |
| **Tốc độ và hiệu năng tính toán** | ⭐⭐☆☆☆ | Độ trễ còn cao gấp 6 lần ARM, cần các kỹ thuật chưng cất và lập lịch khử nhiễu nhanh. |

### 6.2. Lộ trình thực nghiệm ưu tiên đề xuất (Actionable Roadmap)
1. **Giai đoạn 1 (Ưu tiên cao nhất - Ngắn hạn)**:
   - Triển khai **Thực nghiệm A1 (Entropy-guided Remasking)** để dứt điểm lỗi lặp từ.
   - Bổ sung **Thực nghiệm B3 (Đa dạng hóa độ đo BioBERTScore / ROUGE-L)** vào bảng đánh giá.
2. **Giai đoạn 2 (Trung hạn)**:
   - Áp dụng **Thực nghiệm B1 (DPO / SimPO)** để thu hẹp khoảng cách điểm số câu hỏi mở với LLaVA-Med.
   - Thử nghiệm **Thực nghiệm D1 (Lượng tử hóa 4-bit NF4)** phục vụ triển khai rộng rãi.
3. **Giai đoạn 3 (Dài hạn)**:
   - Phát triển **Thực nghiệm A3 (Consistency Distillation)** và **E1 (3D CT/MRI Multi-slice)** để biến LLaDA-MedV thành hệ thống trợ lý chẩn đoán hình ảnh toàn diện trong bệnh viện.

---
*Tài liệu này được biên soạn phục vụ công tác nghiên cứu, tái hiện và phát triển tiếp nối bài báo LLaDA-MedV.*
