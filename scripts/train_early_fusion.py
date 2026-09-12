#!/usr/bin/env python3
"""
scripts/train_early_fusion.py

Huấn luyện mô hình Early Fusion bằng cách nối đặc trưng:
Acoustic (128-d) + XLS-R (1024-d) = 1152-d.
Mặc định chạy ở chế độ tải từ cache (Smoke test).
"""

import sys
from pathlib import Path
import time
import numpy as np
import argparse

# Đảm bảo UTF-8 output trên Windows console
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_classifier import BaselineClassifier

DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "features_cache"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "early_fusion_logistic_regression.joblib"

def load_and_fuse_cache(split: str, n_samples: int, seed: int, smoke_test: bool) -> tuple[np.ndarray, np.ndarray, list]:
    # Tiền tố của file cache
    ac_prefix = "smoke_" if smoke_test else "base_"
    xl_prefix = "xlsr_smoke_" if smoke_test else "xlsr_base_"
    
    ac_path = DEFAULT_CACHE_DIR / f"{ac_prefix}{split}_{n_samples}_s{seed}.npz"
    xl_path = DEFAULT_CACHE_DIR / f"{xl_prefix}{split}_{n_samples}_s{seed}.npz"
    
    if not ac_path.exists():
        raise FileNotFoundError(f"Không tìm thấy cache Acoustic: {ac_path}")
    if not xl_path.exists():
        raise FileNotFoundError(f"Không tìm thấy cache XLS-R: {xl_path}")
        
    ac_data = np.load(ac_path, allow_pickle=True)
    xl_data = np.load(xl_path, allow_pickle=True)
    
    X_ac = ac_data["X"]
    y_ac = ac_data["y"]
    meta_ac = ac_data["meta"]
    
    X_xl = xl_data["X"]
    
    # Kểm tra đồng nhất dữ liệu
    assert len(X_ac) == len(X_xl), f"Số lượng mẫu không khớp: {len(X_ac)} vs {len(X_xl)}"
    assert np.array_equal(y_ac, xl_data["y"]), "Nhãn (Labels) không khớp giữa 2 file cache!"
    
    # Nối 2 vector theo chiều ngang (axis=1) -> 128 + 1024 = 1152
    X_fused = np.hstack((X_ac, X_xl))
    
    return X_fused, y_ac, list(meta_ac)

def evaluate_per_attack(classifier, X, y, meta, threshold=0.5):
    probs = classifier.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    attacks = sorted(list({m["attack_id"] for m in meta if m["attack_id"] != "bonafide"}))
    results = []
    for atk in attacks:
        indices = [i for i, m in enumerate(meta) if m["attack_id"] == atk]
        if not indices: continue
        sub_y = y[indices]
        sub_pred = preds[indices]
        sub_prob = probs[indices]
        tp = np.sum(sub_pred == 1)
        fn = np.sum(sub_pred == 0)
        total = len(indices)
        recall = tp / total if total > 0 else 0.0
        results.append({
            "attack_id": atk,
            "count": total,
            "detected_spoof": int(tp),
            "recall": round(float(recall), 4),
        })
    return results

def run_early_fusion(n_train: int, n_dev: int, n_eval: int, seed: int, smoke_test: bool):
    print("=" * 95)
    mode_str = "SMOKE TEST" if smoke_test else "FULL EXPERIMENT"
    print(f" EARLY FUSION (ACOUSTIC 128-d + XLS-R 1024-d = 1152-d) - ({mode_str})")
    print("=" * 95)
    
    print("[+] Đang tải và ghép nối (Concat) dữ liệu từ Cache...")
    X_train, y_train, meta_train = load_and_fuse_cache("train", n_train, seed, smoke_test)
    X_dev, y_dev, meta_dev = load_and_fuse_cache("dev", n_dev, seed, smoke_test)
    X_eval, y_eval, meta_eval = load_and_fuse_cache("eval", n_eval, seed, smoke_test)
    
    print(f"    - Tập TRAIN : {X_train.shape} (Ghép 128 + 1024 = {X_train.shape[1]})")
    print(f"    - Tập DEV   : {X_dev.shape}")
    print(f"    - Tập EVAL  : {X_eval.shape}")
    
    classifier = BaselineClassifier(class_weight="balanced", random_state=seed, max_iter=2000)
    
    print("\n[+] Đang huấn luyện Logistic Regression trên vector 1152 chiều...")
    t0 = time.time()
    
    # Tạo feature names giả
    f_names = [f"ac_{i}" for i in range(128)] + [f"xlsr_{i}" for i in range(1024)]
    classifier.fit(X_train, y_train, feature_names=f_names)
    
    print(f"    -> Huấn luyện thành công trong {time.time() - t0:.3f}s")
    
    # Đánh giá
    dev_res = classifier.evaluate(X_dev, y_dev, threshold=0.5)
    dev_eer_thresh = dev_res["eer_threshold"]
    dev_tuned = classifier.evaluate(X_dev, y_dev, threshold=dev_eer_thresh)
    eval_tuned = classifier.evaluate(X_eval, y_eval, threshold=dev_eer_thresh)
    
    print("\n" + "=" * 95)
    print(" BẢNG ĐÁNH GIÁ EARLY FUSION 1152-d (KNOWN VS UNSEEN)")
    print("=" * 95)
    print(f"{'Nhóm Tấn Công':<25} {'Tập Dữ Liệu':<14} {'F1-Score':<11} {'ROC-AUC':<11} {'EER':<10}")
    print("-" * 95)
    print(f"{'KNOWN (A01 - A06)':<25} {'DEV':<14} {dev_tuned['f1']:<11.4f} {dev_tuned['roc_auc']:<11.4f} {dev_tuned['eer']:<10.4f}")
    print(f"{'UNSEEN (A07 - A19)':<25} {'EVAL':<14} {eval_tuned['f1']:<11.4f} {eval_tuned['roc_auc']:<11.4f} {eval_tuned['eer']:<10.4f}")
    
    print("\n" + "=" * 95)
    print(" CHI TIẾT TỪNG LOẠI TẤN CÔNG (EVAL UNSEEN)")
    print("=" * 95)
    per_atk = evaluate_per_attack(classifier, X_eval, y_eval, meta_eval, threshold=dev_eer_thresh)
    for res in per_atk:
        print(f"Attack {res['attack_id']}: Recall = {res['recall']*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Chạy chế độ smoke test")
    parser.add_argument("--n-train", type=int, default=2500)
    parser.add_argument("--n-dev", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=2500)
    args = parser.parse_args()
    
    n_tr = 250 if args.smoke_test else args.n_train
    n_de = 100 if args.smoke_test else args.n_dev
    n_ev = 150 if args.smoke_test else args.n_eval
    
    try:
        run_early_fusion(n_train=n_tr, n_dev=n_de, n_eval=n_ev, seed=42, smoke_test=args.smoke_test)
    except Exception as e:
        print(f"Lỗi: {e}")
