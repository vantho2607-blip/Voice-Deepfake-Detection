# STEP 3 AUDIT REPORT — DAP391m: BỘ LỌC 1 (PROBABILITY CLASSIFIER BASELINE)

## 1. Status
**PASS**

Hệ thống phân loại xác suất cơ bản (Probability Baseline Classifier) đã được triển khai, kiểm thử tự động, huấn luyện và đánh giá thực tế thành công trên tập dữ liệu benchmark ASVspoof 2019 Logical Access (LA).

---

## 2. Architecture

Luồng xử lý từ đầu đến cuối tuân thủ nghiêm ngặt pipeline chuẩn:

```text
SQLite Metadata (metadata.db)
  ↓
Audio Path
  ↓
Load Audio & Ensure 16 kHz (FLAC/WAV)
  ↓
Silence Trimming (Leading & Trailing via Short-Time Energy)
  ↓
Amplitude Normalization (Peak to 0.95, Zero-safe)
  ↓
Spectral Subtraction Denoising (STFT -> 10th-percentile Noise Estimation -> Subtraction -> ISTFT)
  ↓
Acoustic Feature Extraction (MFCC, Deltas, Spectral, Time-Frequency Pooling -> 128 dims)
  ↓
StandardScaler (Fit CHỈ trên tập TRAIN)
  ↓
Logistic Regression (class_weight='balanced', L-BFGS, Seed=42)
  ↓
Probability Output [P(bonafide), P(spoof)]
  ↓
Decision (Bonafide vs Spoof dựa theo Decision Threshold / EER Threshold)
```

---

## 3. Dataset

Dữ liệu được truy vấn động từ [metadata.db](file:///D:/Download/DAP391m/data/processed/metadata.db) với chiến lược lấy mẫu phân tầng (Stratified Sampling):

| Split | Tổng số mẫu thực nghiệm | Số mẫu Bonafide | Số mẫu Spoof | Hệ thống tấn công tham gia |
| :--- | :---: | :---: | :---: | :--- |
| **TRAIN** | 2,496 | 300 (12.02%) | 2,196 (87.98%) | A01 - A06 (Known Attacks) |
| **DEV** | 996 | 120 (12.05%) | 876 (87.95%) | A01 - A06 (Known Attacks) |
| **EVAL** | 2,497 | 300 (12.01%) | 2,197 (87.99%) | A07 - A19 (Unseen Attacks) |

* **Quy tắc bảo mật dữ liệu**: Tập **EVAL** hoàn toàn độc lập; không được dùng để huấn luyện, không dùng để fit scaler, và không dùng để chọn threshold.

---

## 4. Features

* **Bộ trích xuất**: `AcousticFeatureExtractor` (`src/features/audio_features.py`)
* **Số chiều cố định**: **128 chiều**
* **Danh mục đặc trưng**:
  1. **MFCC** (20 hệ số): Mean (20) + Std (20) = **40 chiều**
  2. **Delta MFCC** (đạo hàm bậc 1, 20 hệ số): Mean (20) + Std (20) = **40 chiều**
  3. **Delta-Delta MFCC** (đạo hàm bậc 2, 20 hệ số): Mean (20) + Std (20) = **40 chiều**
  4. **Spectral Centroid** (Trọng tâm phổ): Mean (1) + Std (1) = **2 chiều**
  5. **Spectral Rolloff** (Tần số cuộn phổ 85% năng lượng): Mean (1) + Std (1) = **2 chiều**
  6. **Zero-Crossing Rate** (Tỷ lệ đổi dấu thời gian): Mean (1) + Std (1) = **2 chiều**
  7. **RMS Energy** (Năng lượng hiệu dụng khung): Mean (1) + Std (1) = **2 chiều**
* **Kiểm tra an toàn**: 100% không chứa NaN, không chứa Inf, an toàn với tín hiệu im lặng.

---

## 5. Model

* **Classifier**: `sklearn.linear_model.LogisticRegression`
  * Solver: `lbfgs`
  * Max Iterations: `1,000`
  * Random State: `42`
* **Class Balancing**: `class_weight='balanced'` (Xử lý chênh lệch tự nhiên 1:9 giữa Bonafide và Spoof trong benchmark ASVspoof)
* **Standardization**: `sklearn.preprocessing.StandardScaler` (Fit **strictly** chỉ trên tập TRAIN; tập DEV và EVAL chỉ transform dựa theo mean và std của TRAIN)
* **Checkpoint**: [data/processed/models/baseline_logistic_regression.joblib](file:///D:/Download/DAP391m/data/processed/models/baseline_logistic_regression.joblib)

---

## 6. Probability

Mô hình xuất ra phân phối xác suất hợp lệ qua hàm Sigmoid của Logistic Regression:
$$P(\text{bonafide}) + P(\text{spoof}) = 1.0, \quad 0 \le P \le 1$$

Ví dụ dự đoán thực tế từ script `predict_audio.py`:
* Mẫu Bonafide (`LA_T_1138215.flac`):
  * $P(\text{bonafide}) = 0.9693$ ($96.93\%$)
  * $P(\text{spoof}) = 0.0307$ ($3.07\%$)
  * Kết luận: **BONAFIDE**
* Mẫu Spoof A01 (`LA_T_1004644.flac`):
  * $P(\text{bonafide}) = 0.0002$ ($0.02\%$)
  * $P(\text{spoof}) = 0.9998$ ($99.98\%$)
  * Kết luận: **SPOOF**

---

## 7. Overall Results

Kết quả đo đạc trên tập validation (DEV) và kiểm thử mù (EVAL):

| Metric | DEV (Ngưỡng 0.50) | DEV (Ngưỡng EER = 0.80) | EVAL (Ngưỡng 0.50) | EVAL (Ngưỡng Opt DEV = 0.80) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 87.55% | 80.52% | 71.17% | 58.23% |
| **Precision** | 94.03% | 96.84% | 93.62% | 95.50% |
| **Recall** | 91.67% | 80.48% | 72.14% | 55.12% |
| **F1-Score** | **0.9283** | **0.8791** | **0.8149** | **0.6990** |
| **ROC-AUC** | **0.8746** | **0.8746** | **0.7604** | **0.7604** |
| **Equal Error Rate (EER)** | **19.34%** | **19.34%** | **32.39%** | **32.39%** |

### Ma trận nhầm lẫn (Confusion Matrix) trên EVAL (ngưỡng 0.80):
* **True Negative** (Bonafide đúng): **243** mẫu
* **False Positive** (Bonafide $\to$ Spoof - False Alarm): **57** mẫu (False Alarm Rate: 19.00%)
* **False Negative** (Spoof $\to$ Bonafide - Miss): **986** mẫu (Miss Rate: 44.88%)
* **True Positive** (Spoof đúng): **1,211** mẫu

---

## 8. Known vs Unseen

| Nhóm Tấn Công | Tập Dữ Liệu | F1-Score | ROC-AUC | EER |
| :--- | :---: | :---: | :---: | :---: |
| **KNOWN (A01 - A06)** | **DEV** | **0.8791** | **0.8746** | **19.34%** |
| **UNSEEN (A07 - A19)** | **EVAL** | **0.6990** | **0.7604** | **32.39%** |

> [!IMPORTANT]
> **Phân tích học thuật**: Hiệu năng mô hình giảm đáng kể khi chuyển từ Known Attacks ($EER = 19.34\%$) sang Unseen Attacks ($EER = 32.39\%$). Đây là hiện tượng chuẩn được ghi nhận trong tất cả các bài báo nghiên cứu về ASVspoof 2019: các mô hình phân loại tuyến tính trên đặc trưng âm học truyền thống không thể tổng quát hóa hoàn hảo cho các công nghệ tổng hợp giọng nói chưa từng xuất hiện trong tập huấn luyện. Hiện tượng này chứng minh sự cần thiết của **BƯỚC 4 (Acoustic CoT)** và **BƯỚC 5 (Continual Learning)**.

---

## 9. Per-Attack Results

Chi tiết tỷ lệ phát hiện trên từng hệ thống tấn công chưa thấy (Unseen Attacks A07 - A19) trong tập EVAL:

| Attack ID | Công nghệ sinh giọng | Số mẫu | Phát hiện đúng (TP) | Bỏ sót (FN) | Detection Recall | Mean $P(\text{spoof})$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **A07** | TTS Vocoder + GAN | 169 | 48 | 121 | 28.40% | 0.5169 |
| **A08** | TTS Neural waveform | 169 | 153 | 16 | **90.53%** | **0.9318** |
| **A09** | TTS Vocoder | 169 | 166 | 3 | **98.22%** | **0.9885** |
| **A10** | TTS Neural waveform | 169 | 40 | 129 | 23.67% | 0.4833 |
| **A11** | TTS Griffin-Lim | 169 | 96 | 73 | 56.80% | 0.6999 |
| **A12** | TTS Neural waveform | 169 | 36 | 133 | 21.30% | 0.4958 |
| **A13** | TTS_VC Concat + Filtering | 169 | 58 | 111 | 34.32% | 0.5733 |
| **A14** | TTS_VC Vocoder | 169 | 114 | 55 | 67.46% | 0.7606 |
| **A15** | TTS_VC Neural waveform | 169 | 157 | 12 | **92.90%** | **0.9490** |
| **A16** | TTS Waveform Concat | 169 | 48 | 121 | 28.40% | 0.5020 |
| **A17** | VC Waveform Filtering | 169 | 134 | 35 | **79.29%** | **0.8509** |
| **A18** | VC Vocoder | 169 | 56 | 113 | 33.14% | 0.5502 |
| **A19** | VC Spectral Filtering | 169 | 105 | 64 | 62.13% | 0.7618 |

---

## 10. Leakage Audit

* [x] **TRAIN không overlap DEV**: `overlap = 0`
* [x] **TRAIN không overlap EVAL**: `overlap = 0`
* [x] **DEV không overlap EVAL**: `overlap = 0`
* [x] **Speaker overlap với EVAL**: `overlap = 0` (67 speakers của EVAL hoàn toàn độc lập với 20 speakers của Train/Dev)
* [x] **StandardScaler chỉ fit trên TRAIN**: Khẳng định qua pipeline architecture (`pipeline.fit(X_train, y_train)`).
* [x] **Classifier chỉ train trên TRAIN**: Không có bất kỳ mẫu DEV hoặc EVAL nào trong `fit()`.
* [x] **Ngưỡng quyết định không chọn từ EVAL**: Ngưỡng $0.80$ được tính từ EER của tập DEV.
* [x] **Không dùng nhãn attack của EVAL để tối ưu model**: Nhãn attack chỉ dùng sau cùng để sinh bảng thống kê.
* **Kết luận**: **100% PASS (Leak-Free)**.

---

## 11. Tests

* **Test Suite**: `pytest tests -v`
  * `tests/test_step1_metadata.py`: **10 / 10 PASSED**
  * `tests/test_step2_preprocessing.py`: **12 / 12 PASSED**
  * `tests/test_step3_classifier.py`: **10 / 10 PASSED**
* **Tổng cộng**: **32 / 32 PASSED (100%)** trong 2.07 giây.

---

## 12. Inference

* **Script**: [scripts/predict_audio.py](file:///D:/Download/DAP391m/scripts/predict_audio.py)
* **Kiểm chứng thực tế**:
  * Chạy trên file Bonafide `LA_T_1138215.flac` $\to$ Trả về `BONAFIDE` (Confidence 96.93%).
  * Chạy trên file Spoof A01 `LA_T_1004644.flac` $\to$ Trả về `SPOOF` (Confidence 99.98%).
* **Kết luận**: **PASS**.

---

## 13. Limitations

1. **Đặc trưng âm học thủ công (Handcrafted Features)**: 128 đặc trưng (MFCC, Deltas, Spectral, Energy) bắt được các đặc điểm âm học tĩnh và động cơ bản, nhưng khó nắm bắt được các artifact tinh vi sinh ra từ các mô hình tổng hợp neural hiện đại (A10, A12).
2. **Mô hình tuyến tính (Linear Classifier)**: Logistic Regression là mô hình tuyến tính đơn giản, có tính giải thích cao và huấn luyện nhanh, nhưng giới hạn khả năng học các mối quan hệ phi tuyến phức tạp trong không gian đặc trưng âm thanh.
3. **Hiện tượng suy giảm trên Unseen Attacks**: Tỷ lệ EER tăng từ $19.34\%$ lên $32.39\%$ khi đối mặt với các công nghệ tấn công chưa từng gặp.

---

## 14. Future Work

* **BƯỚC 4**: Nghiên cứu Acoustic Chain-of-Thought (Acoustic CoT) và thử thách đa chiều (Multi-dimensional stress testing) để giải thích sâu về mặt ngữ âm học và âm thanh học đối với các trường hợp bị bỏ sót.
* **BƯỚC 5**: Triển khai Continual Learning trên nền tảng Classifier checkpoint này để liên tục thích nghi với các cuộc tấn công mới (A07 - A19) mà không bị quên kiến thức trên các cuộc tấn công cũ (A01 - A06).
* **BƯỚC 6**: Đóng gói hoàn chỉnh hệ thống và nghiệm thu đồ án.

*(Tuân thủ nguyên tắc: Không tự ý triển khai các bước trên).*
