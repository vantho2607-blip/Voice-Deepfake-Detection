#!/usr/bin/env python3
"""
scripts/evaluate_baseline.py

Script đánh giá độc lập mô hình Baseline Classifier đã huấn luyện trên các tập DEV và EVAL:
  - Tải checkpoint từ data/processed/models/baseline_logistic_regression.joblib.
  - Đánh giá Accuracy, Precision, Recall, F1, ROC-AUC, EER.
  - In ma trận nhầm lẫn (Confusion Matrix).
  - Phân tích chi tiết theo từng loại tấn công (A07 - A19 trên EVAL).
  - So sánh Known attacks (DEV) vs Unseen attacks (EVAL).
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
from scripts.train_baseline import (
    evaluate_per_attack,
    extract_features_for_samples,
    query_samples_by_split,
)

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.db"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "baseline_logistic_regression.joblib"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "features_cache"


def run_evaluation(
    db_path: Path,
    model_path: Path,
    n_dev: int = 1000,
    n_eval: int = 2500,
    seed: int = 42,
    threshold: float = None,
) -> None:
    print("=" * 95)
    print(" ĐÁNH GIÁ ĐỘC LẬP MÔ HÌNH BASELINE CLASSIFIER (BƯỚC 3)")
    print("=" * 95)
    print(f"[*] Database Path   : {db_path}")
    print(f"[*] Model Checkpoint: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint không tồn tại: {model_path}. Hãy chạy train_baseline.py trước.")

    classifier = BaselineClassifier.load(model_path)
    preprocessor = AudioPreprocessor(target_sr=16000, norm_target_peak=0.95)
    extractor = AcousticFeatureExtractor(sr=16000, n_fft=512, hop_length=256, n_mels=20)

    # 1. Lấy dữ liệu DEV và EVAL
    print("\n[+] Đang chuẩn bị dữ liệu DEV và EVAL...")
    dev_samples = query_samples_by_split(db_path, "dev", n_dev, seed=seed)
    eval_samples = query_samples_by_split(db_path, "eval", n_eval, seed=seed)

    cache_dev = DEFAULT_CACHE_DIR / f"base_dev_{n_dev}_s{seed}.npz"
    cache_eval = DEFAULT_CACHE_DIR / f"base_eval_{n_eval}_s{seed}.npz"

    X_dev, y_dev, meta_dev = extract_features_for_samples(dev_samples, preprocessor, extractor, cache_dev)
    X_eval, y_eval, meta_eval = extract_features_for_samples(eval_samples, preprocessor, extractor, cache_eval)

    # 2. Xác định ngưỡng tối ưu từ DEV
    dev_metrics_default = classifier.evaluate(X_dev, y_dev, threshold=0.5)
    opt_threshold = threshold if threshold is not None else dev_metrics_default["eer_threshold"]
    dev_metrics_tuned = classifier.evaluate(X_dev, y_dev, threshold=opt_threshold)

    # 3. Đánh giá EVAL với ngưỡng tối ưu từ DEV (Tuyệt đối không tune trên EVAL)
    eval_metrics_default = classifier.evaluate(X_eval, y_eval, threshold=0.5)
    eval_metrics_tuned = classifier.evaluate(X_eval, y_eval, threshold=opt_threshold)

    # 4. In bảng kết quả
    print("\n" + "=" * 95)
    print(" BẢNG CHỈ SỐ ĐÁNH GIÁ TỔNG HỢP (SUMMARY METRICS)")
    print("=" * 95)
    print(f"{'Split':<8} {'Threshold':<12} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<10} {'ROC-AUC':<9} {'EER':<8}")
    print("-" * 95)
    print(f"{'DEV':<8} {'0.50 (def)':<12} {dev_metrics_default['accuracy']:<10.4f} {dev_metrics_default['precision']:<11.4f} {dev_metrics_default['recall']:<9.4f} {dev_metrics_default['f1']:<10.4f} {dev_metrics_default['roc_auc']:<9.4f} {dev_metrics_default['eer']:<8.4f}")
    print(f"{'DEV':<8} {f'{opt_threshold:.2f} (EER)':<12} {dev_metrics_tuned['accuracy']:<10.4f} {dev_metrics_tuned['precision']:<11.4f} {dev_metrics_tuned['recall']:<9.4f} {dev_metrics_tuned['f1']:<10.4f} {dev_metrics_tuned['roc_auc']:<9.4f} {dev_metrics_tuned['eer']:<8.4f}")
    print(f"{'EVAL':<8} {'0.50 (def)':<12} {eval_metrics_default['accuracy']:<10.4f} {eval_metrics_default['precision']:<11.4f} {eval_metrics_default['recall']:<9.4f} {eval_metrics_default['f1']:<10.4f} {eval_metrics_default['roc_auc']:<9.4f} {eval_metrics_default['eer']:<8.4f}")
    print(f"{'EVAL':<8} {f'{opt_threshold:.2f} (opt)':<12} {eval_metrics_tuned['accuracy']:<10.4f} {eval_metrics_tuned['precision']:<11.4f} {eval_metrics_tuned['recall']:<9.4f} {eval_metrics_tuned['f1']:<10.4f} {eval_metrics_tuned['roc_auc']:<9.4f} {eval_metrics_tuned['eer']:<8.4f}")

    print("\n" + "=" * 95)
    print(" MA TRẬN NHẦM LẪN TRÊN TẬP EVAL (CONFUSION MATRIX)")
    print("=" * 95)
    cm = eval_metrics_tuned["confusion_matrix"]
    print(f"  * True Negative  (Bonafide đúng)  : {cm['tn']:,}")
    print(f"  * False Positive (Bonafide -> Spoof): {cm['fp']:,} (False Alarm Rate: {eval_metrics_tuned['far_bonafide_as_spoof']*100:.2f}%)")
    print(f"  * False Negative (Spoof -> Bonafide): {cm['fn']:,} (Miss Rate: {eval_metrics_tuned['frr_spoof_as_bonafide']*100:.2f}%)")
    print(f"  * True Positive  (Spoof đúng)     : {cm['tp']:,}")

    # 5. Đánh giá chi tiết từng Attack trên EVAL
    per_attack = evaluate_per_attack(classifier, X_eval, y_eval, meta_eval, threshold=opt_threshold)
    print("\n" + "=" * 95)
    print(" BẢNG CHI TIẾT TỪNG LOẠI TẤN CÔNG UNSEEN TRÊN EVAL (PER-ATTACK EVALUATION)")
    print("=" * 95)
    print(f"{'Attack':<8} {'Số mẫu':<10} {'Phát hiện đúng':<16} {'Bỏ sót (Miss)':<16} {'Detection Recall':<18} {'Mean P(spoof)':<15}")
    print("-" * 95)
    for atk in per_attack:
        print(f"{atk['attack_id']:<8} {atk['count']:<10} {atk['detected_spoof']:<16} {atk['missed_as_bonafide']:<16} {atk['recall']:<18.4f} {atk['mean_prob_spoof']:<15.4f}")


def main():
    parser = argparse.ArgumentParser(description="Đánh giá độc lập mô hình Baseline Classifier")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="Đường dẫn SQLite database")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Đường dẫn checkpoint model")
    parser.add_argument("--n-dev", type=int, default=1000, help="Số mẫu dev")
    parser.add_argument("--n-eval", type=int, default=2500, help="Số mẫu eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threshold", type=float, default=None, help="Ngưỡng phân loại tùy chọn (mặc định lấy từ DEV EER)")
    args = parser.parse_args()

    run_evaluation(
        db_path=Path(args.db_path),
        model_path=Path(args.model_path),
        n_dev=args.n_dev,
        n_eval=args.n_eval,
        seed=args.seed,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
