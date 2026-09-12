#!/usr/bin/env python3
"""
scripts/train_baseline.py

Huấn luyện và đánh giá mô hình Baseline Classifier (BƯỚC 3):
Luồng thực thi:
  1. Lấy dữ liệu từ SQLite (data/processed/metadata.db).
  2. Stratified Sampling theo đúng chuẩn ASVspoof 2019:
     - Train: A01 - A06 + Bonafide
     - Dev: A01 - A06 + Bonafide
     - Eval: A07 - A19 + Bonafide (Hoàn toàn độc lập, không dùng để train)
  3. Chạy qua pipeline BƯỚC 2 (Ensure 16kHz, Trim Silence, Peak Norm, Spectral Subtraction).
  4. Trích xuất 128 đặc trưng âm học (MFCC, Deltas, Spectral, Pooling).
  5. Fit StandardScaler và LogisticRegression (class_weight='balanced') CHỈ trên TRAIN.
  6. Đánh giá trên DEV (Accuracy, Precision, Recall, F1, ROC-AUC, EER).
  7. Đánh giá trên EVAL (Đánh giá cuối cùng, Per-Attack Analysis, Known vs Unseen).
  8. Kiểm tra rò rỉ dữ liệu (Data Leakage Audit).
  9. Lưu checkpoint vào data/processed/models/baseline_logistic_regression.joblib.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import random
import sqlite3
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

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

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.db"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "baseline_logistic_regression.joblib"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "features_cache"


def query_samples_by_split(
    db_path: Path, split: str, n_samples: int, seed: int = 42
) -> List[Dict[str, str]]:
    """
    Truy vấn và lấy mẫu đại diện có phân tầng (Stratified Sampling) từ SQLite.
    Bảo đảm cân bằng giữa bonafide (~10-15%) và phân bổ đều giữa các hệ thống tấn công spoof.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Lấy tất cả mẫu trong split
    cursor.execute("""
    SELECT file_id, speaker_id, split, label, COALESCE(attack_id, 'bonafide'), audio_path
    FROM samples
    WHERE split = ?;
    """, (split,))
    rows = cursor.fetchall()
    conn.close()

    # Gom nhóm theo attack_type (hoặc bonafide)
    grouped = defaultdict(list)
    for r in rows:
        item = {
            "file_id": r[0],
            "speaker_id": r[1],
            "split": r[2],
            "label": r[3],
            "attack_id": r[4],
            "audio_path": r[5],
        }
        grouped[item["attack_id"]].append(item)

    rng = random.Random(seed)

    # Tỷ lệ bonafide mục tiêu ~12%
    n_bonafide = max(10, int(n_samples * 0.12))
    n_spoof = n_samples - n_bonafide

    selected = []
    # 1. Lấy bonafide
    bonafide_pool = grouped.get("bonafide", [])
    rng.shuffle(bonafide_pool)
    selected.extend(bonafide_pool[:n_bonafide])

    # 2. Lấy spoof phân bổ đều giữa các attack systems
    spoof_keys = [k for k in grouped.keys() if k != "bonafide"]
    if spoof_keys:
        per_attack = max(1, n_spoof // len(spoof_keys))
        for k in sorted(spoof_keys):
            pool = grouped[k]
            rng.shuffle(pool)
            selected.extend(pool[:per_attack])

    # Shuffle tổng thể
    rng.shuffle(selected)
    return selected[:n_samples]


def extract_features_for_samples(
    samples: List[Dict[str, str]],
    preprocessor: AudioPreprocessor,
    extractor: AcousticFeatureExtractor,
    cache_path: Path = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    """
    Trích xuất đặc trưng với pipeline BƯỚC 2:
      Load -> Ensure 16kHz -> Silence Trim -> Peak Norm -> Spectral Subtraction -> Feature Extractor
    Có hỗ trợ cache dạng npz để tăng tốc thực nghiệm.
    """
    if cache_path and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], list(data["meta"])

    X_list = []
    y_list = []
    meta_list = []

    for idx, s in enumerate(samples):
        raw_path = Path(s["audio_path"])
        # Pipeline BƯỚC 2:
        res = preprocessor.process_file(raw_path, output_dir=None)
        clean_waveform = res["processed_waveform"]
        sr = res["target_sr"]

        # Pipeline BƯỚC 3: Trích xuất đặc trưng âm học
        feats = extractor.extract(clean_waveform)
        label_int = 1 if s["label"] == "spoof" else 0

        X_list.append(feats)
        y_list.append(label_int)
        meta_list.append(s)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, meta=meta_list)

    return X, y, meta_list


def audit_data_leakage(
    train_samples: List[Dict[str, str]],
    dev_samples: List[Dict[str, str]],
    eval_samples: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Kiểm tra rò rỉ dữ liệu (Data Leakage Audit):
      - Không trùng lặp file_id giữa Train, Dev và Eval.
      - Không trùng lặp speaker_id giữa Train/Dev và Eval.
    """
    train_files = {s["file_id"] for s in train_samples}
    dev_files = {s["file_id"] for s in dev_samples}
    eval_files = {s["file_id"] for s in eval_samples}

    train_spks = {s["speaker_id"] for s in train_samples}
    dev_spks = {s["speaker_id"] for s in dev_samples}
    eval_spks = {s["speaker_id"] for s in eval_samples}

    overlap_train_dev = train_files.intersection(dev_files)
    overlap_train_eval = train_files.intersection(eval_files)
    overlap_dev_eval = dev_files.intersection(eval_files)
    spk_overlap_eval = (train_spks | dev_spks).intersection(eval_spks)

    is_leak_free = (
        len(overlap_train_dev) == 0
        and len(overlap_train_eval) == 0
        and len(overlap_dev_eval) == 0
        and len(spk_overlap_eval) == 0
    )

    return {
        "is_leak_free": is_leak_free,
        "overlap_train_dev_files": len(overlap_train_dev),
        "overlap_train_eval_files": len(overlap_train_eval),
        "overlap_dev_eval_files": len(overlap_dev_eval),
        "speaker_overlap_train_dev_with_eval": len(spk_overlap_eval),
    }


def evaluate_per_attack(
    classifier: BaselineClassifier,
    X: np.ndarray,
    y: np.ndarray,
    meta: List[Dict[str, str]],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Đánh giá chi tiết hiệu năng theo từng loại tấn công (Per-Attack Analysis).
    """
    probs = classifier.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    attacks = sorted(list({m["attack_id"] for m in meta if m["attack_id"] != "bonafide"}))
    results = []

    for atk in attacks:
        indices = [i for i, m in enumerate(meta) if m["attack_id"] == atk]
        if not indices:
            continue
        sub_y = y[indices]
        sub_pred = preds[indices]
        sub_prob = probs[indices]

        # Mẫu attack đều có label=1 (spoof)
        tp = np.sum(sub_pred == 1)
        fn = np.sum(sub_pred == 0)
        total = len(indices)
        recall = tp / total if total > 0 else 0.0
        mean_prob_spoof = float(np.mean(sub_prob))

        results.append({
            "attack_id": atk,
            "count": total,
            "detected_spoof": int(tp),
            "missed_as_bonafide": int(fn),
            "recall": round(float(recall), 4),
            "mean_prob_spoof": round(mean_prob_spoof, 4),
        })

    return results


def run_training_and_evaluation(
    db_path: Path,
    model_path: Path,
    n_train: int,
    n_dev: int,
    n_eval: int,
    seed: int,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    print("=" * 95)
    mode_str = "SMOKE TEST" if smoke_test else "BASELINE EXPERIMENT"
    print(f" BƯỚC 3: HUẤN LUYỆN & ĐÁNH GIÁ BỘ PHÂN LOẠI BASELINE ({mode_str})")
    print("=" * 95)
    print(f"[*] SQLite Database : {db_path}")
    print(f"[*] Model Checkpoint: {model_path}")
    print(f"[*] Cấu hình mẫu     : Train={n_train:,} | Dev={n_dev:,} | Eval={n_eval:,} (Seed={seed})")

    # 1. Khởi tạo components
    preprocessor = AudioPreprocessor(target_sr=16000, norm_target_peak=0.95)
    extractor = AcousticFeatureExtractor(sr=16000, n_fft=512, hop_length=256, n_mels=20)
    classifier = BaselineClassifier(class_weight="balanced", random_state=seed, max_iter=1000)

    # 2. Lấy mẫu phân tầng từ SQLite
    print("\n[+] Đang truy vấn mẫu phân tầng từ SQLite metadata.db...")
    train_samples = query_samples_by_split(db_path, "train", n_train, seed=seed)
    dev_samples = query_samples_by_split(db_path, "dev", n_dev, seed=seed)
    eval_samples = query_samples_by_split(db_path, "eval", n_eval, seed=seed)

    print(f"    - Tập TRAIN: {len(train_samples):,} mẫu (Bonafide: {sum(1 for s in train_samples if s['label']=='bonafide')}, Spoof: {sum(1 for s in train_samples if s['label']=='spoof')})")
    print(f"    - Tập DEV  : {len(dev_samples):,} mẫu (Bonafide: {sum(1 for s in dev_samples if s['label']=='bonafide')}, Spoof: {sum(1 for s in dev_samples if s['label']=='spoof')})")
    print(f"    - Tập EVAL : {len(eval_samples):,} mẫu (Bonafide: {sum(1 for s in eval_samples if s['label']=='bonafide')}, Spoof: {sum(1 for s in eval_samples if s['label']=='spoof')})")

    # 3. Data Leakage Audit
    leakage = audit_data_leakage(train_samples, dev_samples, eval_samples)
    print("\n[+] Kiểm tra rò rỉ dữ liệu (Data Leakage Audit):")
    print(f"    - Overlap file giữa Train & Dev : {leakage['overlap_train_dev_files']}")
    print(f"    - Overlap file giữa Train & Eval: {leakage['overlap_train_eval_files']}")
    print(f"    - Overlap file giữa Dev & Eval  : {leakage['overlap_dev_eval_files']}")
    print(f"    - Overlap speaker với Eval      : {leakage['speaker_overlap_train_dev_with_eval']}")
    if not leakage["is_leak_free"]:
        raise RuntimeError("PHÁT HIỆN RÒ RỈ DỮ LIỆU GIỮA CÁC TẬP DATASET!")
    print("    -> KẾT QUẢ: 100% LEAK-FREE (Dữ liệu hoàn toàn độc lập)")

    # 4. Trích xuất đặc trưng với pipeline BƯỚC 2
    prefix = "smoke_" if smoke_test else "base_"
    cache_train = DEFAULT_CACHE_DIR / f"{prefix}train_{n_train}_s{seed}.npz"
    cache_dev = DEFAULT_CACHE_DIR / f"{prefix}dev_{n_dev}_s{seed}.npz"
    cache_eval = DEFAULT_CACHE_DIR / f"{prefix}eval_{n_eval}_s{seed}.npz"

    print("\n[+] Đang thực thi tiền xử lý BƯỚC 2 & Trích xuất đặc trưng (128 dims)...")
    t0 = time.time()
    X_train, y_train, meta_train = extract_features_for_samples(train_samples, preprocessor, extractor, cache_train)
    X_dev, y_dev, meta_dev = extract_features_for_samples(dev_samples, preprocessor, extractor, cache_dev)
    X_eval, y_eval, meta_eval = extract_features_for_samples(eval_samples, preprocessor, extractor, cache_eval)
    feat_time = time.time() - t0
    print(f"    -> Đã trích xuất xong đặc trưng: {feat_time:.2f}s (Ma trận X_train: {X_train.shape})")

    # 5. Huấn luyện Classifier (Scaler + LogisticRegression trên TRAIN)
    print("\n[+] Đang huấn luyện Logistic Regression (StandardScaler fit CHỈ trên TRAIN)...")
    t_train_start = time.time()
    classifier.fit(X_train, y_train, feature_names=extractor.feature_names)
    train_time = time.time() - t_train_start
    print(f"    -> Huấn luyện thành công trong {train_time:.3f}s")

    # 6. Đánh giá trên DEV để tìm EER threshold
    dev_eval_default = classifier.evaluate(X_dev, y_dev, threshold=0.5)
    dev_eer_thresh = dev_eval_default["eer_threshold"]
    dev_eval_tuned = classifier.evaluate(X_dev, y_dev, threshold=dev_eer_thresh)

    # 7. Đánh giá cuối cùng trên EVAL (Dùng threshold tối ưu từ DEV và default 0.5)
    eval_res_default = classifier.evaluate(X_eval, y_eval, threshold=0.5)
    eval_res_tuned = classifier.evaluate(X_eval, y_eval, threshold=dev_eer_thresh)

    # 8. Đánh giá chi tiết từng Attack System trên EVAL
    per_attack_eval = evaluate_per_attack(classifier, X_eval, y_eval, meta_eval, threshold=dev_eer_thresh)

    # 9. Đánh giá Known vs Unseen
    # Known: DEV (A01-A06) | Unseen: EVAL (A07-A19)
    known_eval = dev_eval_tuned
    unseen_eval = eval_res_tuned

    # 10. Lưu checkpoint
    model_path.parent.mkdir(parents=True, exist_ok=True)
    extra_meta = {
        "n_train": n_train,
        "n_dev": n_dev,
        "n_eval": n_eval,
        "seed": seed,
        "smoke_test": smoke_test,
        "dev_eer": dev_eval_default["eer"],
        "dev_eer_threshold": dev_eer_thresh,
        "eval_eer": eval_res_default["eer"],
        "feature_dim": extractor.feature_dim,
    }
    saved_p = classifier.save(model_path, extra_meta=extra_meta)
    print(f"\n[+] Đã lưu checkpoint mô hình tại: {saved_p}")

    # In kết quả tổng hợp
    print("\n" + "=" * 95)
    print(" BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (OVERALL METRICS)")
    print("=" * 95)
    header = f"{'Split':<8} {'Threshold':<11} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<10} {'ROC-AUC':<9} {'EER':<8}"
    print(header)
    print("-" * 95)
    print(f"{'DEV':<8} {'0.50 (def)':<11} {dev_eval_default['accuracy']:<10.4f} {dev_eval_default['precision']:<11.4f} {dev_eval_default['recall']:<9.4f} {dev_eval_default['f1']:<10.4f} {dev_eval_default['roc_auc']:<9.4f} {dev_eval_default['eer']:<8.4f}")
    print(f"{'DEV':<8} {f'{dev_eer_thresh:.2f} (EER)':<11} {dev_eval_tuned['accuracy']:<10.4f} {dev_eval_tuned['precision']:<11.4f} {dev_eval_tuned['recall']:<9.4f} {dev_eval_tuned['f1']:<10.4f} {dev_eval_tuned['roc_auc']:<9.4f} {dev_eval_tuned['eer']:<8.4f}")
    print(f"{'EVAL':<8} {'0.50 (def)':<11} {eval_res_default['accuracy']:<10.4f} {eval_res_default['precision']:<11.4f} {eval_res_default['recall']:<9.4f} {eval_res_default['f1']:<10.4f} {eval_res_default['roc_auc']:<9.4f} {eval_res_default['eer']:<8.4f}")
    print(f"{'EVAL':<8} {f'{dev_eer_thresh:.2f} (opt)':<11} {eval_res_tuned['accuracy']:<10.4f} {eval_res_tuned['precision']:<11.4f} {eval_res_tuned['recall']:<9.4f} {eval_res_tuned['f1']:<10.4f} {eval_res_tuned['roc_auc']:<9.4f} {eval_res_tuned['eer']:<8.4f}")

    print("\n" + "=" * 95)
    print(" BẢNG ĐÁNH GIÁ KNOWN VS UNSEEN ATTACKS")
    print("=" * 95)
    print(f"{'Nhóm Tấn Công':<25} {'Tập Dữ Liệu':<14} {'F1-Score':<11} {'ROC-AUC':<11} {'EER':<10}")
    print("-" * 95)
    print(f"{'KNOWN (A01 - A06)':<25} {'DEV':<14} {known_eval['f1']:<11.4f} {known_eval['roc_auc']:<11.4f} {known_eval['eer']:<10.4f}")
    print(f"{'UNSEEN (A07 - A19)':<25} {'EVAL':<14} {unseen_eval['f1']:<11.4f} {unseen_eval['roc_auc']:<11.4f} {unseen_eval['eer']:<10.4f}")

    print("\n" + "=" * 95)
    print(" BẢNG CHI TIẾT TỪNG LOẠI TẤN CÔNG TRÊN EVAL (PER-ATTACK EVALUATION)")
    print("=" * 95)
    print(f"{'Attack':<8} {'Số mẫu':<10} {'Phát hiện đúng':<16} {'Bỏ sót (Miss)':<16} {'Detection Recall':<18} {'Mean P(spoof)':<15}")
    print("-" * 95)
    for atk in per_attack_eval:
        print(f"{atk['attack_id']:<8} {atk['count']:<10} {atk['detected_spoof']:<16} {atk['missed_as_bonafide']:<16} {atk['recall']:<18.4f} {atk['mean_prob_spoof']:<15.4f}")

    return {
        "dev_default": dev_eval_default,
        "dev_tuned": dev_eval_tuned,
        "eval_default": eval_res_default,
        "eval_tuned": eval_res_tuned,
        "per_attack": per_attack_eval,
        "leakage": leakage,
    }


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá Baseline Classifier cho ASVspoof 2019 LA")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="Đường dẫn SQLite database")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Đường dẫn lưu model checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Chạy chế độ smoke test kiểm tra nhanh")
    parser.add_argument("--n-train", type=int, default=2500, help="Số mẫu huấn luyện (mặc định: 2500)")
    parser.add_argument("--n-dev", type=int, default=1000, help="Số mẫu dev (mặc định: 1000)")
    parser.add_argument("--n-eval", type=int, default=2500, help="Số mẫu eval (mặc định: 2500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (mặc định: 42)")
    args = parser.parse_args()

    n_tr = 250 if args.smoke_test else args.n_train
    n_de = 100 if args.smoke_test else args.n_dev
    n_ev = 150 if args.smoke_test else args.n_eval

    run_training_and_evaluation(
        db_path=Path(args.db_path),
        model_path=Path(args.model_path),
        n_train=n_tr,
        n_dev=n_de,
        n_eval=n_ev,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
