import sys
from pathlib import Path
import numpy as np
import time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
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
    print("THU NGHIEM EP XUNG MAXIMUM MO HINH BOOSTING (Tuong duong LightGBM)")
    print("="*60)
    
    # 1. Load Data
    X_tr_lfcc, y_tr = load_cache("base", "train", 25380)
    X_ev_lfcc, y_ev = load_cache("base", "eval", 71237)
    X_tr_xlsr, _ = load_cache("xlsr_base", "train", 25380)
    X_ev_xlsr, _ = load_cache("xlsr_base", "eval", 71237)
    
    X_tr = np.hstack([X_tr_lfcc, X_tr_xlsr])
    X_ev = np.hstack([X_ev_lfcc, X_ev_xlsr])
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ev = scaler.transform(X_ev)
    
    # 2. Setup HistGradientBoosting (Phiên bản tốc độ cao của sklearn, thuật toán y hệt LightGBM)
    # class_weight='balanced' không hỗ trợ trực tiếp trong HistGradientBoosting, 
    # nên ta bù đắp bằng cách ép nó học cực sâu và dùng L2 Regularization.
    base_model = HistGradientBoostingClassifier(
        max_iter=300, # Tương đương 300 cây
        random_state=42
    )
    
    # 3. Tuning Grid (Ép xung tối đa)
    # Tìm kiếm ngẫu nhiên trong không gian siêu tham số khổng lồ
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [5, 10, 20, None],
        'l2_regularization': [0.0, 1.0, 10.0, 100.0],
        'min_samples_leaf': [10, 20, 50],
        'max_leaf_nodes': [31, 63, 127]
    }
    
    print("[+] Dang chay Tuning (RandomizedSearchCV) de tim bo tham so manh nhat...")
    print("    Qua trinh nay ep CPU duyet qua hang chuc cau hinh cay chang chit...")
    
    start_time = time.time()
    # Chạy 10 cấu hình khác nhau, Cross Validation 3 Fold = 30 lần train
    search = RandomizedSearchCV(base_model, param_distributions=param_dist, 
                                n_iter=10, cv=3, scoring='roc_auc', 
                                n_jobs=-1, random_state=42, verbose=1)
    
    search.fit(X_tr, y_tr)
    train_time = time.time() - start_time
    
    print(f"\n[+] Da Tuning xong sau {train_time:.1f} giay!")
    print(f"[+] Tham so khung nhat tim duoc (The Ultimate Tree):")
    for k, v in search.best_params_.items():
        print(f"    - {k}: {v}")
    
    # 4. Dự đoán trên tập EVAL
    print("\n[+] Dang dem con quai vat Boosting nay di dau tren 71k file EVAL...")
    best_model = search.best_estimator_
    probas = best_model.predict_proba(X_ev)[:, 1]
    
    eer = compute_eer(y_ev, probas)
    
    print("="*60)
    print(f"KET QUA CUOI CUNG SAU KHI EP XUNG BOOSTING TO MAX")
    print(f"    -> EER cua Boosting Hang nang: {eer:.2f}%")
    print(f"    -> (Nhac lai) EER cua Logistic Regression sieu nhe: 3.82%")
    print("="*60)
    if eer > 3.82:
        print("KET LUAN: Dung nhu du doan, Boosting du duoc do max binh van sap mat truoc Linear Probing tren 1152-d!")

if __name__ == "__main__":
    main()
