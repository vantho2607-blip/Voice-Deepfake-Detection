import sys
from pathlib import Path
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_cache(prefix: str, split: str, n_samples: int, seed: int = 42):
    cache_path = PROJECT_ROOT / "data" / "processed" / "features_cache" / f"{prefix}_{split}_{n_samples}_s{seed}.npz"
    data = np.load(cache_path, allow_pickle=True)
    return data['X'], data['y']

def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer * 100

def main():
    print("="*60)
    print("DAI CHIEN DUONG THANG: SVM vs LOGISTIC REGRESSION")
    print("="*60)
    
    # Load Data
    X_tr_lfcc, y_tr = load_cache("base", "train", 25380)
    X_ev_lfcc, y_ev = load_cache("base", "eval", 71237)
    X_tr_xlsr, _ = load_cache("xlsr_base", "train", 25380)
    X_ev_xlsr, _ = load_cache("xlsr_base", "eval", 71237)
    
    X_tr = np.hstack([X_tr_lfcc, X_tr_xlsr])
    X_ev = np.hstack([X_ev_lfcc, X_ev_xlsr])
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ev = scaler.transform(X_ev)
    
    # 1. Logistic Regression
    print("[1] Dang train Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_tr, y_tr)
    probas_lr = lr.predict_proba(X_ev)[:, 1]
    eer_lr = compute_eer(y_ev, probas_lr)
    print(f"    -> EER cua Logistic Regression: {eer_lr:.2f}%\n")
    
    # 2. Linear SVM
    print("[2] Dang train Linear SVM (Ho tro Vector Machine)...")
    # LinearSVC không xuất ra xác suất (predict_proba), nó chỉ xuất ra khoảng cách.
    # Nên để tính EER công bằng, ta dùng khoảng cách tới Decision Function.
    svm = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42, dual=False)
    svm.fit(X_tr, y_tr)
    
    # Dùng decision_function thay vì predict_proba
    scores_svm = svm.decision_function(X_ev)
    eer_svm = compute_eer(y_ev, scores_svm)
    print(f"    -> EER cua Linear SVM: {eer_svm:.2f}%\n")
    
    print("="*60)
    if eer_svm < eer_lr:
        print(f"KET LUAN: SVM THANG! No chem duong thang xin hon ({eer_svm:.2f}% vs {eer_lr:.2f}%)")
    else:
        print(f"KET LUAN: LOGISTIC REGRESSION THANG! Xac suat cua no van trum ({eer_lr:.2f}% vs {eer_svm:.2f}%)")

if __name__ == "__main__":
    main()
