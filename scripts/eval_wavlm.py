import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_cache(prefix: str, split: str, n_samples: int, seed: int = 42):
    cache_path = PROJECT_ROOT / "data" / "processed" / "features_cache" / f"{prefix}_{split}_{n_samples}.npz"
    data = np.load(cache_path, allow_pickle=True)
    return data['X'], data['y']

def load_cache_lfcc(prefix: str, split: str, n_samples: int, seed: int = 42):
    # Dùng lfcc cũ, seed 42
    cache_path = PROJECT_ROOT / "data" / "processed" / "features_cache" / f"{prefix}_{split}_{n_samples}_s{seed}.npz"
    data = np.load(cache_path, allow_pickle=True)
    return data['X']

def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer * 100

def main():
    print("="*60)
    print("DANH GIA SUC MANH CUA WAVLM (vs XLS-R)")
    print("="*60)
    
    # Số lượng bản ghi đã được extract ngẫu nhiên (Giờ là FULL)
    N_TRAIN = 25380
    N_EVAL = 71237
    
    # 1. Load WavLM features
    print("[+] Dang nap dac trung WavLM (768-d)...")
    X_tr_wavlm, y_tr = load_cache("wavlm_base", "train", N_TRAIN)
    X_ev_wavlm, y_ev = load_cache("wavlm_base", "eval", N_EVAL)
    
    # 2. Train and Eval CHỈ với WavLM (để xem bản thân nó mạnh tới đâu)
    print("[+] Dang Benchmark Logistic Regression voi RIENG WavLM (768-d)...")
    scaler = StandardScaler()
    X_tr_w = scaler.fit_transform(X_tr_wavlm)
    X_ev_w = scaler.transform(X_ev_wavlm)
    
    lr_w = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_w.fit(X_tr_w, y_tr)
    probas_w = lr_w.predict_proba(X_ev_w)[:, 1]
    eer_w = compute_eer(y_ev, probas_w)
    
    print("="*60)
    print(f"KET QUA KHI THAY NAO XLS-R BANG WAVLM-BASE (Tren tap con {N_EVAL} test files)")
    print(f" -> EER cua WavLM-Base-Plus: {eer_w:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
