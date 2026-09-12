# Hệ Thống Phát Hiện Giả Mạo Giọng Nói (Voice Anti-Spoofing)
Dự án Nhận diện Deepfake Audio (TTS, Voice Conversion) - Khóa luận/Đồ án Thực hành (DAP391m)

## 📌 Tổng Quan Dự Án
Dự án xây dựng một hệ thống Trí tuệ Nhân tạo đa phương thức (Hybrid AI) để phát hiện các tệp âm thanh giả mạo. Hệ thống được phát triển và đánh giá trên tập dữ liệu chuẩn quốc tế **ASVspoof 2019 Logical Access (LA)**, với tổng cộng hơn 121.000 tệp âm thanh.

## 🏗 Kiến Trúc Hệ Thống (Hybrid Dual-Stream Early Fusion)
Thay vì sử dụng một mô hình hộp đen duy nhất, hệ thống kết hợp sức mạnh của 2 thế giới: Xử lý Tín hiệu Số (DSP) và Học sâu (Deep Learning).

1. **Tiền xử lý tín hiệu (Audio Preprocessing):**
   - Đưa về chuẩn 16kHz.
   - Cắt khoảng lặng hai đầu bằng năng lượng RMS (Silence Trimming).
   - Chuẩn hóa biên độ (Peak Normalization).
   - Khử nhiễu môi trường bằng thuật toán **Spectral Subtraction**.

2. **Trích xuất Đặc trưng Luồng kép (Dual-Stream Feature Extraction):**
   - **Luồng Vật lý (Acoustic - 128 chiều):** Thuật toán LFCC (Linear Frequency Cepstral Coefficients) kết hợp ZCR, RMS, Spectral Centroid để soi các dị thường ở tần số cao.
   - **Luồng AI Ngôn ngữ (Deep SSL - 1024 chiều):** Sử dụng mô hình `wav2vec2-xls-r-300m` (đóng băng/frozen) để trích xuất biểu diễn ngữ điệu sâu.

3. **Thuật toán Phân loại (Backend Classifier):**
   - Áp dụng **Logistic Regression** (Linear Probing) để phân chia mặt phẳng tuyến tính trên không gian siêu chiều (1152 chiều).

## 📊 Báo Cáo Kết Quả Thực Nghiệm (Tập EVAL - 71.237 mẫu)
Hệ thống đã trải qua các đợt thực nghiệm khắc nghiệt trên 13 loại mã độc hoàn toàn chưa từng gặp (Unseen Attacks: A07 - A19).

| Phương Pháp / Cấu hình | Thuật toán Phân loại | DEV EER (Known) | EVAL EER (Unseen) | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Acoustic LFCC (128-d)** | Logistic Regression | 18.64% | 28.36% | 79.41% |
| **Deep XLS-R (1024-d)** | XGBoost | 1.14% | 7.98% | 97.54% |
| **Deep XLS-R (1024-d)** | Logistic Regression | 0.42% | 3.96% | 99.24% |
| **Early Fusion (1152-d)** | **Logistic Regression** | **0.71%** | **3.82%** | **99.31%** |

*Kết luận:* Phương pháp Early Fusion (Kết hợp LFCC + XLS-R) đạt kỷ lục EER **3.82%**, khẳng định vị thế ưu việt của mô hình Lai.

## 🛠 Cấu Trúc Mã Nguồn
- `src/`: Lõi thuật toán (Tiền xử lý, Trích xuất đặc trưng LFCC/XLS-R, Wrapper Mô hình).
- `scripts/`: Mã thực thi (Xây dựng DB, Huấn luyện Early Fusion, Đánh giá baseline).
- `data/`: Lưu trữ dữ liệu gốc, cơ sở dữ liệu SQLite và bộ đệm (Cache) numpy.

## 🚀 Hướng Dẫn Sử Dụng
1. Cài đặt các thư viện yêu cầu: `pip install -r requirements.txt`
2. Chạy `scripts/build_metadata_db.py` để đóng gói cơ sở dữ liệu.
3. Chạy `scripts/train_early_fusion.py` để huấn luyện và tái tạo lại kỷ lục EER 3.82%.
