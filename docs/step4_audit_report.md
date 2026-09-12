# BÁO CÁO NGHIỆM THU BƯỚC 4 — ACOUSTIC CHAIN OF EVIDENCE (ACoE)
## Hệ Thống Phát Hiện Voice Cloning Nâng Cao — DAP391m

---

## 1. Trạng Thái (Status)
**PASS (100% HOÀN THÀNH VÀ ĐẠT TẤT CẢ TIÊU CHUẨN KIỂM ĐỊNH)**

BƯỚC 4 — **Acoustic Chain of Evidence (ACoE)** đã được thiết kế, triển khai mã nguồn, tích hợp trơn tru với mô hình Step 3, vượt qua toàn bộ 47/47 ca kiểm thử tự động, và hoàn thành thực nghiệm đánh giá đa chiều trên 480 mẫu âm thanh thuộc toàn bộ 19 họ thuật toán tấn công (A01 — A19) của benchmark chuẩn quốc tế ASVspoof 2019 Logical Access (LA).

---

## 2. Định Nghĩa & Phạm Vi Nghiêm Ngặt Của Bước 4

### 2.1 Định Nghĩa Chuỗi Chứng Cứ Âm Học (ACoE)
ACoE **không phải là một mô hình phân loại mới**, không phải deep learning, không phải LLM sinh văn bản ngẫu nhiên. ACoE là một **lớp cấu trúc chứng cứ số hóa tất định (Deterministic Evidence Layer)** giúp:
1. Trích xuất đa chiều các đặc trưng vật lý và âm học của tín hiệu tiếng nói (Phổ tần, Thời gian, Cepstral, Ngữ điệu/Cao độ).
2. Kiểm tra độ ổn định và độ tin cậy của quyết định phân loại (từ mô hình Step 3) thông qua các **thử thách âm học có kiểm soát (Controlled Acoustic Challenges)**.
3. Cung cấp số liệu phân tích định lượng làm tiền đề khoa học vững chắc cho **BƯỚC 5 — Continual Learning**.

### 2.2 Các Nguyên Tắc Bắt Buộc Đã Tuân Thủ Tuyệt Đối
* [x] **Giữ nguyên Checkpoint Step 3**: Sử dụng nguyên vẹn checkpoint `data/processed/models/baseline_logistic_regression.joblib`, không huấn luyện lại hay thay đổi cấu trúc 128 đặc trưng của Step 3.
* [x] **Bảo vệ toàn vẹn dữ liệu gốc**: 100% file âm thanh tại `data/archive/LA/LA/` được giữ nguyên vẹn (được kiểm chứng qua SHA-256 hash và modification time).
* [x] **Tính tất định & Không chế tạo dữ liệu ảo**: Mọi tính toán F0, spectral, temporal đều dựa trên công thức toán học; khi tín hiệu im lặng hoặc không có tuần hoàn, trường prosodic được trả về an toàn với `status: "unavailable"` và `None`.
* [x] **Không nhảy cóc**: Dừng lại hoàn toàn sau khi hoàn thiện Bước 4; không tự ý triển khai Bước 5 (Continual Learning).

---

## 3. Kiến Trúc Hệ Thống (Architecture & Pipeline)

Luồng phân tích ACoE diễn ra hoàn toàn trong bộ nhớ RAM:

```text
Audio File (.flac / .wav)
  │
  ▼
[BƯỚC 2] Tiền xử lý âm thanh (AudioPreprocessor)
  ├─ Resample về chuẩn 16 kHz
  ├─ Cắt tỉa khoảng lặng (Leading & Trailing Silence Trimming)
  ├─ Chuẩn hóa biên độ (Peak Normalization to 0.95)
  └─ Lọc nhiễu phổ thích ứng (Spectral Subtraction)
  │
  ▼
Clean Waveform (16 kHz, RAM)
  ├──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  ▼                                                                  ▼
[BƯỚC 3 Baseline Classifier]                        [ACoE Multi-Dimensional Evidence Extraction]
  ├─ Trích xuất 128 acoustic features                ├─ Phổ tần (Spectral): Centroid, Rolloff,
  ├─ StandardScaler (Fit TRAIN)                      │   Bandwidth, Flatness, Spectral Flux
  └─ Logistic Regression                             ├─ Thời gian (Temporal): RMS Energy,
     └─ P(bonafide), P(spoof)                        │   RMS Variation, ZCR, Silence/Voiced ratio
                                                     ├─ Cepstral: MFCC 20, Deltas, Delta-Deltas
                                                     └─ Ngữ điệu (Prosodic): F0 Tracking (Autocorr)
                                                         (Trả về an toàn nếu im lặng)
  │                                                                  │
  └───────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
             [Controlled Acoustic Challenges & Stress-Testing]
               ├─ Gain 0.90x (-0.92 dB)
               ├─ Gain 1.10x (+0.83 dB)
               ├─ Additive White Gaussian Noise (SNR = 35 dB, Seed cố định)
               └─ Resampling Perturbation (16 kHz -> 15.2 kHz -> 16 kHz)
                                  │
                                  ▼
             [Acoustic Consistency Score Calculation & Schema Export]
               ├─ Điểm nhất quán âm học tổng hợp [0.0, 1.0]
               ├─ Báo cáo trực quan Terminal / Bảng tổng hợp
               └─ Xuất file cấu trúc chuẩn JSON (AcousticChainOfEvidence)
```

---

## 4. Các Nhóm Chứng Cứ Âm Học Trong ACoE (Evidence Dimensions)

ACoE phân rã một mẫu âm thanh thành 5 chiều chứng cứ khách quan:

1. **Classifier Evidence**:
   - Xác suất hậu nghiệm $P(\text{bonafide})$ và $P(\text{spoof})$ từ Step 3.
   - Quyết định phân loại cơ bản (`bonafide` hoặc `spoof`).
2. **Spectral Evidence (Chứng cứ phổ tần)**:
   - **Spectral Centroid**: Trọng tâm phổ tần năng lượng (Hz).
   - **Spectral Rolloff (85%)**: Tần số mà 85% năng lượng tập trung bên dưới.
   - **Spectral Bandwidth**: Độ rộng dải phổ quanh trọng tâm.
   - **Spectral Flatness**: Tỷ số giữa trung bình hình học và trung bình số học của phổ công suất (đo độ mịn/nhiễu của âm).
   - **Spectral Flux**: Tốc độ biến thiên giữa các khung phổ liên tiếp.
3. **Temporal Evidence (Chứng cứ miền thời gian)**:
   - **RMS Energy & RMS Variation**: Năng lượng hiệu dụng và độ biến thiên biên độ.
   - **Zero-Crossing Rate (ZCR)**: Tần suất đổi dấu tín hiệu.
   - **Silence Ratio**: Tỷ lệ khung năng lượng thấp (< -40 dBFS).
   - **Voiced Ratio Estimate**: Tỷ lệ khung có năng lượng hữu thanh.
4. **Cepstral Evidence (Chứng cứ Cepstral)**:
   - Vector trung bình và độ lệch chuẩn của 20 dải Mel-Frequency Cepstral Coefficients (MFCC), Delta MFCC và Delta-Delta MFCC.
5. **Prosodic Evidence (Chứng cứ ngữ điệu / Cao độ $F_0$)**:
   - Sử dụng giải thuật **Normalized Autocorrelation F0 Tracker** trên cửa sổ 40 ms, bước nhảy 10 ms trong dải tần số giọng người [60 Hz, 400 Hz].
   - **Xử lý biên nghiêm ngặt**: Nếu tín hiệu im lặng hoặc có ít hơn 5 khung hữu thanh tin cậy ($r_{\text{peak}} < 0.35$), hệ thống tự động gán `status: "unavailable"` và giữ các trường $F_0$ ở giá trị `None`, không bao giờ bịa đặt giá trị số.

---

## 5. Thử Thách Âm Học Có Kiểm Soát & Điểm Nhất Quán (Stability Analysis)

### 5.1 Bốn Thử Thách Âm Học Thực Tế
Để kiểm tra xem quyết định của bộ phân loại có nhạy cảm thái quá trước các biến đổi môi trường thông thường hay không, ACoE áp dụng 4 phép biến đổi nhẹ có kiểm soát:
1. **Gain 0.90x**: Giảm âm lượng nhẹ (-0.92 dB).
2. **Gain 1.10x**: Tăng âm lượng nhẹ (+0.83 dB).
3. **Additive Noise (SNR = 35 dB)**: Thêm nhiễu trắng ngẫu nhiên với tỷ số tín hiệu trên nhiễu cao (35 dB) với seed cố định = 42, mô phỏng nhiễu đường truyền phòng thu.
4. **Resampling (15.2 kHz)**: Resample đa thức $16\,\text{kHz} \to 15.2\,\text{kHz} \to 16\,\text{kHz}$ mô phỏng hiện tượng jitter trong codec viễn thông.

### 5.2 Công Thức Điểm Nhất Quán Âm Học (Acoustic Consistency Score)
Điểm số được định nghĩa theo hàm số tất định và bị chặn nghiêm ngặt trong khoảng $[0.0, 1.0]$:

$$\text{Consistency Score} = \max\left(0.0, \, \min\left(1.0, \, 1.0 - \overline{|\Delta P_{\text{spoof}}|} - 0.2 \cdot \overline{\Delta_{\text{evidence}}}\right)\right)$$

Trong đó:
- $\overline{|\Delta P_{\text{spoof}}|}$: Trung bình độ dịch chuyển tuyệt đối của xác suất spoof qua các thử thách.
- $\overline{\Delta_{\text{evidence}}}$: Độ lệch tương đối của vector đặc trưng âm học trước và sau biến đổi:
  $$\Delta_{\text{evidence}} = \frac{\| \mathbf{x}_{\text{perturbed}} - \mathbf{x}_{\text{orig}} \|_2}{\| \mathbf{x}_{\text{orig}} \|_2 + 10^{-8}}$$

> [!NOTE]
> **Ý nghĩa thực tế**: Điểm số $\ge 0.90$ thể hiện mô hình giữ vững quyết định và đặc trưng âm học ít bị lay chuyển trước các nhiễu loạn nhẹ. Nếu điểm số tụt xuống dưới $0.80$, mẫu âm thanh nằm ở vùng ranh giới quyết định mong manh (vùng quyết định không ổn định).

---

## 6. Kết Quả Thực Nghiệm Trên 480 Mẫu (Known vs Unseen Attacks)

Được thực thi qua script tự động [scripts/run_acoustic_challenges.py](file:///D:/Download/DAP391m/scripts/run_acoustic_challenges.py) lấy mẫu phân tầng ngẫu nhiên trên toàn bộ cơ sở dữ liệu `metadata.db`:

### Bảng 1: So sánh tổng quan giữa các nhóm âm thanh

| Nhóm Âm Thanh | Phân Tập | Số lượng | Tỷ lệ nhận diện đúng | Mean $P(\text{spoof})$ | Điểm nhất quán âm học | Mean $\|\Delta P\|$ | Mean Feature $\Delta$ | Flip Rate (Đổi nhãn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bonafide (Thật)** | Dev | 50 | 52.00% | 0.4797 | 0.9126 | 0.0769 | 0.0524 | 8.5% |
| **Bonafide (Thật)** | Eval | 50 | 66.00% | 0.4149 | 0.9020 | 0.0850 | 0.0653 | 15.0% |
| **Known Attacks (A01-A06)** | Dev | 120 | **93.33%** | **0.8908** | **0.9638** | **0.0234** | **0.0640** | **2.3%** |
| **Unseen Attacks (A07-A19)**| Eval | 260 | **72.31%** | **0.7005** | **0.9374** | **0.0525** | **0.0503** | **5.9%** |

### Nhận định khoa học cốt lõi:
1. **Độ ổn định vượt trội trên Known Attacks**: Các thuật toán đã học (A01 - A06) đạt độ chính xác **93.33%**, xác suất dự đoán $P(\text{spoof})$ rất dứt khoát ($0.8908$), và điểm nhất quán rất cao ($0.9638$). Khi bị thử thách âm học, tỷ lệ đổi quyết định (flip rate) chỉ là **2.3%**.
2. **Sự suy giảm rõ rệt trên Unseen Attacks**: Khi đối mặt với 13 thuật toán sinh giọng mới lạ (A07 - A19), tỷ lệ phát hiện tụt giảm xuống **72.31%** (giảm 21%), giá trị trung bình $P(\text{spoof})$ trôi về vùng lưỡng lự ($0.7005$), và tỷ lệ bị đổi nhãn dưới tác động thử thách tăng lên **5.9%**.

---

## 7. Bảng Chi Tiết Từng Thuật Toán Tấn Công (A01 — A19)

| Mã Attack | Phân Loại | Công Nghệ Sinh Giọng | Số mẫu | Detection Recall | Mean $P(\text{spoof})$ | Điểm nhất quán | Phân Tích Hiện Tượng |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **A01** | Known | Neural Vocoder | 20 | **100.00%** | 0.9686 | 0.9753 | Nhận diện hoàn hảo, điểm ổn định cao |
| **A02** | Known | Vocoder | 20 | **100.00%** | 0.9964 | 0.9876 | Artifact phổ rõ rệt, mô hình tự tin tuyệt đối |
| **A03** | Known | Waveform Filtering | 20 | **95.00%** | 0.9141 | 0.9665 | Tín hiệu lọc dải cao để lại dấu vết năng lượng |
| **A04** | Known | Waveform Splicing | 20 | **75.00%** | 0.6606 | 0.9180 | Ghép nối âm học tự nhiên hơn, khó nhận diện hơn |
| **A05** | Known | Neural Waveform | 20 | **100.00%** | 0.9711 | 0.9781 | Dấu vân vocoder nhân tạo rất mạnh |
| **A06** | Known | Spectral Filtering | 20 | **90.00%** | 0.8339 | 0.9577 | Nhận diện tốt |
| **A07** | **Unseen** | TTS Vocoder + GAN | 20 | **40.00%** | 0.4358 | 0.9225 | **Bỏ sót nghiêm trọng**: GAN làm mượt artifact phổ |
| **A08** | **Unseen** | TTS Neural Waveform | 20 | **95.00%** | 0.9135 | 0.9740 | Nhận diện tốt nhờ đặc trưng pha/cepstral |
| **A09** | **Unseen** | TTS Vocoder | 20 | **100.00%** | 0.9991 | 0.9853 | Vocoder cổ điển để lại méo hài âm lớn |
| **A10** | **Unseen** | TTS Neural Waveform | 20 | **45.00%** | 0.4737 | 0.9103 | **Bỏ sót nghiêm trọng**: Waveform tự nhiên cao |
| **A11** | **Unseen** | TTS Griffin-Lim | 20 | **60.00%** | 0.6083 | 0.9105 | Khôi phục pha không hoàn hảo nhưng khó bắt |
| **A12** | **Unseen** | TTS Neural Waveform | 20 | **65.00%** | 0.5586 | 0.9268 | Ngưỡng quyết định nằm sát 0.5 |
| **A13** | **Unseen** | TTS_VC Concat+Filtering| 20 | **75.00%** | 0.7195 | 0.9412 | Chuyển đổi giọng ghép nối |
| **A14** | **Unseen** | TTS_VC Vocoder | 20 | **85.00%** | 0.7728 | 0.9501 | Dấu hiệu vocoder nhận biết được |
| **A15** | **Unseen** | TTS_VC Neural Waveform | 20 | **100.00%** | 0.9780 | 0.9796 | Mô hình phát hiện hoàn toàn |
| **A16** | **Unseen** | TTS Waveform Concat | 20 | **60.00%** | 0.5700 | 0.9141 | Mối nối sóng âm bị che giấu |
| **A17** | **Unseen** | VC Waveform Filtering | 20 | **95.00%** | 0.8959 | 0.9618 | Bộ lọc tạo dị thường năng lượng |
| **A18** | **Unseen** | VC Vocoder | 20 | **40.00%** | 0.4538 | 0.8854 | **Bỏ sót nghiêm trọng**: Voice Conversion chất lượng cao |
| **A19** | **Unseen** | VC Spectral Filtering | 20 | **80.00%** | 0.7269 | 0.9250 | Nhận diện ở mức chấp nhận được |

---

## 8. Phân Tích Tác Động Của Từng Thử Thách Âm Học

| Thử Thách | Biến Đổi | Mean $\|\Delta P(\text{spoof})\|$ | Mean Relative Feature $\Delta$ | Tỷ Lệ Đảo Quyết Định (Flip Rate) |
| :--- | :--- | :---: | :---: | :---: |
| **Gain 0.90x** | Giảm biên độ -0.92 dB | 0.0217 | 0.0003 | 2.92% |
| **Gain 1.10x** | Tăng biên độ +0.83 dB | 0.0196 | 0.0003 | 2.29% |
| **Additive Noise** | Nhiễu trắng Gaussian SNR = 35 dB | **0.1273** | **0.1792** | **16.04%** |
| **Resampling** | 16 kHz $\to$ 15.2 kHz $\to$ 16 kHz | 0.0359 | 0.0423 | 3.54% |

### Giải thích kỹ thuật:
- **Khả năng chịu đựng Gain & Resampling rất tốt**: Bộ chuẩn hóa biên độ ở Bước 2 giúp mô hình hầu như miễn nhiễm với thay đổi độ to ($|\Delta P| \approx 0.02$, flip rate $< 3\%$). Resampling gây biến đổi tần số nhẹ chỉ làm lay chuyển $3.54\%$ số mẫu.
- **Điểm yếu trước nhiễu dải cao (High-frequency Noise Vulnerability)**: Additive noise ở SNR 35 dB tác động trực tiếp lên các hệ số MFCC bậc cao và Spectral Flatness ($\Delta_{\text{feature}} = 17.92\%$), làm dịch chuyển xác suất trung bình $0.1273$ và gây lật nhãn $16.04\%$ số mẫu. Đây là phát hiện tối quan trọng để tăng cường độ bền vững (robustness) trong Continual Learning ở Bước 5.

---

## 9. Bộ Biểu Đồ Minh Họa (Visual Figures in `docs/figures/`)

Tất cả 5 biểu đồ báo cáo đã được kết xuất ở độ phân giải 300 DPI:

1. **[Figure 1: `docs/figures/p_spoof_distribution.png`](file:///D:/Download/DAP391m/docs/figures/p_spoof_distribution.png)**: Phân bố xác suất $P(\text{spoof})$ theo dạng boxplot cho 3 nhóm (Bonafide, Known A01-A06, Unseen A07-A19).
2. **[Figure 2: `docs/figures/consistency_score_distribution.png`](file:///D:/Download/DAP391m/docs/figures/consistency_score_distribution.png)**: Phân bố Acoustic Consistency Score cho thấy sự ổn định cao trên Known Attacks và độ phân tán lớn hơn trên Unseen Attacks.
3. **[Figure 3: `docs/figures/known_vs_unseen_comparison.png`](file:///D:/Download/DAP391m/docs/figures/known_vs_unseen_comparison.png)**: Biểu đồ cột ghép so sánh Recall (93.3% vs 72.3%), Mean $P(\text{spoof})$ (89.1% vs 70.1%), và Consistency Score (96.4% vs 93.7%).
4. **[Figure 4: `docs/figures/per_attack_recall.png`](file:///D:/Download/DAP391m/docs/figures/per_attack_recall.png)**: Biểu đồ cột thể hiện tỷ lệ phát hiện chi tiết từ A01 đến A19, tô màu phân biệt rạch ròi giữa Known (xanh dương) và Unseen (đỏ), làm lộ rõ các "tử huyệt" A07, A10, A18.
5. **[Figure 5: `docs/figures/challenge_delta_p_spoof.png`](file:///D:/Download/DAP391m/docs/figures/challenge_delta_p_spoof.png)**: So sánh mức độ tác động của 4 loại thử thách lên độ dịch chuyển xác suất và độ lệch đặc trưng.

---

## 10. Kiểm Thử Tự Động Toàn Diện (Test Suite Audit)

Lệnh thực thi:
```bash
py -3.12 -m pytest tests -v
```

Kết quả: **47 / 47 PASSED (100%) trong 2.25 giây**.

Danh sách các nhóm kiểm thử:
- `tests/test_step1_metadata.py`: **10 / 10 PASSED** (Kiểm tra cơ sở dữ liệu SQLite, integrity, count khớp protocol).
- `tests/test_step2_preprocessing.py`: **12 / 12 PASSED** (Resampling, trimming, normalization, spectral subtraction, safety guards).
- `tests/test_step3_classifier.py`: **10 / 10 PASSED** (Feature 128 chiều, Logistic Regression, probability bounds, model save/load).
- `tests/test_step4_acoustic.py`: **15 / 15 PASSED**:
  - `test_evidence_structure_and_json_serialization`: PASS.
  - `test_prosodic_unavailable_behavior`: PASS.
  - `test_gain_challenge`: PASS.
  - `test_additive_noise_challenge_reproducibility`: PASS.
  - `test_additive_noise_on_silence`: PASS.
  - `test_resampling_challenge`: PASS.
  - `test_resampling_on_empty`: PASS.
  - `test_runner_execution_and_score_bounds`: PASS.
  - `test_prosodic_pitch_tracking_on_clean_tone`: PASS (bắt chính xác 200 Hz tone).
  - `test_prosodic_on_silence_returns_unavailable`: PASS (trả về unavailable an toàn).
  - `test_spectral_evidence_properties`: PASS.
  - `test_temporal_evidence_properties`: PASS.
  - `test_cepstral_evidence_dimensions`: PASS (20 chiều MFCC/Deltas).
  - `test_analyze_waveform_in_ram`: PASS.
  - `test_real_audio_file_unmodified_guard`: PASS (**Tuyệt đối không chạm vào file gốc: SHA-256 hash và mtime bất biến**).

---

## 11. Hướng Dẫn Sử Dụng (CLI Guide)

### 11.1 Phân tích ACoE cho một file âm thanh đơn lẻ
```bash
py -3.12 scripts/analyze_acoustic.py --audio "data/archive/LA/LA/ASVspoof2019_LA_train/flac/LA_T_1138215.flac"
```

Xuất kết quả chi tiết ra JSON:
```bash
py -3.12 scripts/analyze_acoustic.py --audio "path/to/audio.flac" --output "output_acoe.json"
```

### 11.2 Chạy lại thử thách âm học hàng loạt & tự động vẽ biểu đồ
```bash
py -3.12 scripts/run_acoustic_challenges.py --samples-per-attack 20 --bonafide-samples 50
```

---

## 12. Ý Nghĩa Tiền Đề Cho BƯỚC 5 — Continual Learning

Kết quả của Bước 4 đã cung cấp bằng chứng thực nghiệm rõ ràng:
1. **Sự suy giảm nghiêm trọng trên các cuộc tấn công chưa từng gặp (A07, A10, A18)** chứng minh rằng mô hình tĩnh (Static Baseline Classifier) sẽ nhanh chóng bị qua mặt trong môi trường thực tế khi kẻ tấn công nâng cấp công nghệ tổng hợp giọng nói.
2. **Khả năng phân biệt qua Consistency Score**: Các mẫu âm thanh bị nhầm lẫn hoặc nằm ở ranh giới quyết định thường có sự biến thiên lớn dưới thử thách nhiễu dải cao.
3. **Mục tiêu của Bước 5 (Continual Learning)** sẽ là:
   - Cho phép mô hình cập nhật và thích ứng với các cuộc tấn công mới (Unseen Attacks) theo từng đợt dữ liệu (streaming / sequential tasks).
   - Ngăn chặn triệt để hiện tượng quên tai hại (**Catastrophic Forgetting**) đối với các cuộc tấn công cũ (A01 - A06) và âm thanh thật (Bonafide).

*(Tuân thủ nguyên tắc: Không tự ý triển khai Bước 5 trước khi có yêu cầu).*
