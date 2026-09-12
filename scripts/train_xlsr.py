#!/usr/bin/env python3
"""
scripts/train_xlsr.py

Huấn luyện và đánh giá bộ phân loại độc lập sử dụng đặc trưng sâu XLS-R (1024-d).
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

from src.features.xlsr_features import XlsrFeatureExtractor
from src.models.baseline_classifier import BaselineClassifier
from src.preprocessing.audio_preprocessor import AudioPreprocessor

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.db"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "xlsr_logistic_regression.joblib"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "features_cache"


def query_samples_by_split(
    db_path: Path, split: str, n_samples: int, seed: int = 42
) -> List[Dict[str, str]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT file_id, speaker_id, split, label, COALESCE(attack_id, 'bonafide'), audio_path
    FROM samples
    WHERE split = ?;
    """, (split,))
    rows = cursor.fetchall()
    conn.close()

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

    n_bonafide = max(10, int(n_samples * 0.12))
    n_spoof = n_samples - n_bonafide

    selected = []
    bonafide_pool = list(grouped.get("bonafide", []))
    rng.shuffle(bonafide_pool)
    selected.extend(bonafide_pool[:n_bonafide])

    spoof_keys = [k for k in grouped.keys() if k != "bonafide"]
    if spoof_keys:
        per_attack = max(1, n_spoof // len(spoof_keys))
        for k in sorted(spoof_keys):
            pool = list(grouped[k])
            rng.shuffle(pool)
            selected.extend(pool[:per_attack])

    rng.shuffle(selected)
    return selected[:n_samples]


def extract_features_for_samples(
    samples: List[Dict[str, str]],
    preprocessor: AudioPreprocessor,
    extractor: XlsrFeatureExtractor,
    cache_path: Path = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    if cache_path and cache_path.exists():
        print(f"    [XLS-R] Tải feature từ cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], list(data["meta"])

    X_list = []
    y_list = []
    meta_list = []
    
    total = len(samples)
    print(f"    [XLS-R] Bắt đầu trích xuất cho {total} mẫu. Quá trình này có thể mất thời gian...")
    
    import torch
    import gc

    for idx, s in enumerate(samples):
        if (idx + 1) % 50 == 0:
            print(f"      ...đã xử lý {idx + 1}/{total} mẫu")
            
        raw_path = Path(s["audio_path"])
        res = preprocessor.process_file(raw_path, output_dir=None)
        clean_waveform = res["processed_waveform"]
        sr = res["target_sr"]

        # Trích xuất XLS-R (1024 dims)
        feats = extractor.extract(clean_waveform, sr=sr)
        label_int = 1 if s["label"] == "spoof" else 0

        X_list.append(feats)
        y_list.append(label_int)
        meta_list.append(s)
        
        if (idx + 1) % 200 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, meta=meta_list)

    return X, y, meta_list


def evaluate_per_attack(
    classifier: BaselineClassifier,
    X: np.ndarray,
    y: np.ndarray,
    meta: List[Dict[str, str]],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
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
    mode_str = "SMOKE TEST" if smoke_test else "XLS-R EXPERIMENT"
    print(f" HUẤN LUYỆN & ĐÁNH GIÁ VỚI ĐẶC TRƯNG SÂU XLS-R (1024-d) - ({mode_str})")
    print("=" * 95)
    print(f"[*] SQLite Database : {db_path}")
    print(f"[*] Model Checkpoint: {model_path}")
    print(f"[*] Cấu hình mẫu     : Train={n_train:,} | Dev={n_dev:,} | Eval={n_eval:,} (Seed={seed})")

    preprocessor = AudioPreprocessor(target_sr=16000, norm_target_peak=0.95)
    extractor = XlsrFeatureExtractor()
    classifier = BaselineClassifier(class_weight="balanced", random_state=seed, max_iter=1000)

    print("\n[+] Đang truy vấn mẫu phân tầng từ SQLite metadata.db...")
    train_samples = query_samples_by_split(db_path, "train", n_train, seed=seed)
    dev_samples = query_samples_by_split(db_path, "dev", n_dev, seed=seed)
    eval_samples = query_samples_by_split(db_path, "eval", n_eval, seed=seed)

    print(f"    - Tập TRAIN: {len(train_samples):,} mẫu")
    print(f"    - Tập DEV  : {len(dev_samples):,} mẫu")
    print(f"    - Tập EVAL : {len(eval_samples):,} mẫu")

    prefix = "xlsr_smoke_" if smoke_test else "xlsr_base_"
    cache_train = DEFAULT_CACHE_DIR / f"{prefix}train_{n_train}_s{seed}.npz"
    cache_dev = DEFAULT_CACHE_DIR / f"{prefix}dev_{n_dev}_s{seed}.npz"
    cache_eval = DEFAULT_CACHE_DIR / f"{prefix}eval_{n_eval}_s{seed}.npz"

    print("\n[+] Đang thực thi tiền xử lý BƯỚC 2 & Trích xuất đặc trưng XLS-R (1024 dims)...")
    t0 = time.time()
    X_train, y_train, meta_train = extract_features_for_samples(train_samples, preprocessor, extractor, cache_train)
    X_dev, y_dev, meta_dev = extract_features_for_samples(dev_samples, preprocessor, extractor, cache_dev)
    X_eval, y_eval, meta_eval = extract_features_for_samples(eval_samples, preprocessor, extractor, cache_eval)
    feat_time = time.time() - t0
    print(f"    -> Đã trích xuất xong đặc trưng: {feat_time:.2f}s (Ma trận X_train: {X_train.shape})")

    print("\n[+] Đang huấn luyện Logistic Regression (StandardScaler fit CHỈ trên TRAIN)...")
    t_train_start = time.time()
    classifier.fit(X_train, y_train, feature_names=extractor.feature_names)
    train_time = time.time() - t_train_start
    print(f"    -> Huấn luyện thành công trong {train_time:.3f}s")

    dev_eval_default = classifier.evaluate(X_dev, y_dev, threshold=0.5)
    dev_eer_thresh = dev_eval_default["eer_threshold"]
    dev_eval_tuned = classifier.evaluate(X_dev, y_dev, threshold=dev_eer_thresh)

    eval_res_default = classifier.evaluate(X_eval, y_eval, threshold=0.5)
    eval_res_tuned = classifier.evaluate(X_eval, y_eval, threshold=dev_eer_thresh)

    per_attack_eval = evaluate_per_attack(classifier, X_eval, y_eval, meta_eval, threshold=dev_eer_thresh)

    known_eval = dev_eval_tuned
    unseen_eval = eval_res_tuned

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

    print("\n" + "=" * 95)
    print(" BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (XLS-R 1024-d)")
    print("=" * 95)
    header = f"{'Split':<8} {'Threshold':<11} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<10} {'ROC-AUC':<9} {'EER':<8}"
    print(header)
    print("-" * 95)
    print(f"{'DEV':<8} {'0.50 (def)':<11} {dev_eval_default['accuracy']:<10.4f} {dev_eval_default['precision']:<11.4f} {dev_eval_default['recall']:<9.4f} {dev_eval_default['f1']:<10.4f} {dev_eval_default['roc_auc']:<9.4f} {dev_eval_default['eer']:<8.4f}")
    print(f"{'DEV':<8} {f'{dev_eer_thresh:.2f} (EER)':<11} {dev_eval_tuned['accuracy']:<10.4f} {dev_eval_tuned['precision']:<11.4f} {dev_eval_tuned['recall']:<9.4f} {dev_eval_tuned['f1']:<10.4f} {dev_eval_tuned['roc_auc']:<9.4f} {dev_eval_tuned['eer']:<8.4f}")
    print(f"{'EVAL':<8} {'0.50 (def)':<11} {eval_res_default['accuracy']:<10.4f} {eval_res_default['precision']:<11.4f} {eval_res_default['recall']:<9.4f} {eval_res_default['f1']:<10.4f} {eval_res_default['roc_auc']:<9.4f} {eval_res_default['eer']:<8.4f}")
    print(f"{'EVAL':<8} {f'{dev_eer_thresh:.2f} (opt)':<11} {eval_res_tuned['accuracy']:<10.4f} {eval_res_tuned['precision']:<11.4f} {eval_res_tuned['recall']:<9.4f} {eval_res_tuned['f1']:<10.4f} {eval_res_tuned['roc_auc']:<9.4f} {eval_res_tuned['eer']:<8.4f}")

    print("\n" + "=" * 95)
    print(" BẢNG ĐÁNH GIÁ KNOWN VS UNSEEN ATTACKS (XLS-R 1024-d)")
    print("=" * 95)
    print(f"{'Nhóm Tấn Công':<25} {'Tập Dữ Liệu':<14} {'F1-Score':<11} {'ROC-AUC':<11} {'EER':<10}")
    print("-" * 95)
    print(f"{'KNOWN (A01 - A06)':<25} {'DEV':<14} {known_eval['f1']:<11.4f} {known_eval['roc_auc']:<11.4f} {known_eval['eer']:<10.4f}")
    print(f"{'UNSEEN (A07 - A19)':<25} {'EVAL':<14} {unseen_eval['f1']:<11.4f} {unseen_eval['roc_auc']:<11.4f} {unseen_eval['eer']:<10.4f}")

    return {}


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Baseline Classifier dùng XLS-R 1024-d")
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
