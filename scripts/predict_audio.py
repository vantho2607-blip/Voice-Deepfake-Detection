#!/usr/bin/env python3
"""
scripts/predict_audio.py

Dự đoán xác suất và phân loại cho một file âm thanh đơn lẻ:
Pipeline:
  Audio File -> Preprocessing BƯỚC 2 -> Feature Extractor (128 dims) -> Baseline Classifier -> Probability
"""

import argparse
from pathlib import Path
import sys

# Đảm bảo UTF-8 output trên Windows console
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.audio_features import AcousticFeatureExtractor
from src.models.baseline_classifier import BaselineClassifier
from src.preprocessing.audio_preprocessor import AudioPreprocessor

DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "baseline_logistic_regression.joblib"


def predict_audio(audio_path: Path, model_path: Path, threshold: float = 0.5) -> dict:
    if not audio_path.exists():
        raise FileNotFoundError(f"File audio không tồn tại: {audio_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint không tồn tại: {model_path}. Hãy chạy train_baseline.py trước.")

    # 1. Load mô hình
    classifier = BaselineClassifier.load(model_path)

    # 2. Tiền xử lý BƯỚC 2
    preprocessor = AudioPreprocessor(target_sr=16000, norm_target_peak=0.95)
    res = preprocessor.process_file(audio_path, output_dir=None)
    clean_waveform = res["processed_waveform"]

    # 3. Trích xuất đặc trưng
    extractor = AcousticFeatureExtractor(sr=16000, n_fft=512, hop_length=256, n_mels=20)
    feat_vec = extractor.extract(clean_waveform)

    # 4. Dự đoán xác suất
    result = classifier.predict_single(feat_vec, threshold=threshold)
    result["file"] = audio_path.name
    result["sample_rate"] = res["target_sr"]
    result["duration"] = res["final_duration"]

    return result


def main():
    parser = argparse.ArgumentParser(description="Dự đoán Voice Spoofing / Deepfake cho một file audio")
    parser.add_argument("--audio", type=str, required=True, help="Đường dẫn tới file audio cần kiểm tra")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Đường dẫn model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng phân loại (mặc định 0.5)")
    args = parser.parse_args()

    res = predict_audio(Path(args.audio), Path(args.model), threshold=args.threshold)

    print("=" * 60)
    print(" KẾT QUẢ PHÂN TÍCH VOICE SPOOFING")
    print("=" * 60)
    print(f"File                : {res['file']}")
    print(f"Sample Rate         : {res['sample_rate']} Hz")
    print(f"Duration            : {res['duration']}s")
    print(f"Probability Bonafide: {res['prob_bonafide']:.4f} ({res['prob_bonafide']*100:.2f}%)")
    print(f"Probability Spoof   : {res['prob_spoof']:.4f} ({res['prob_spoof']*100:.2f}%)")
    print(f"Decision Threshold  : {res['threshold']:.2f}")
    print(f"Prediction          : {res['prediction'].upper()}")
    print(f"Confidence          : {res['confidence']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
