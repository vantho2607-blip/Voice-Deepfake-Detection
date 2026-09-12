import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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
    # Tìm điểm mà FNR và FPR gần nhau nhất
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer * 100, auc(fpr, tpr) * 100

def main():
    print("="*60)
    print("BENCHMARK 5 MO HINH HOC MAY TREN TAP EVAL 71K")
    print("="*60)
    
    print("[1/4] Dang nap du lieu Early Fusion (1152-d)...")
    # Tải LFCC
    X_tr_lfcc, y_tr = load_cache("base", "train", 25380)
    X_ev_lfcc, y_ev = load_cache("base", "eval", 71237)
    # Tải XLS-R
    X_tr_xlsr, _ = load_cache("xlsr_base", "train", 25380)
    X_ev_xlsr, _ = load_cache("xlsr_base", "eval", 71237)
    
    # Nối đặc trưng (Early Fusion)
    X_tr = np.hstack([X_tr_lfcc, X_tr_xlsr])
    X_ev = np.hstack([X_ev_lfcc, X_ev_xlsr])
    
    print("[2/4] Chuan hoa du lieu (StandardScaler)...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ev = scaler.transform(X_ev)
    
    # Khai báo 5 mô hình để thi đấu
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Naive Bayes (GNB)": GaussianNB(),
        "Random Forest (Trees)": RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=42)
    }
    
    print("[3/4] Bat dau huan luyen va thi dau...\n")
    
    results = []
    
    for name, model in models.items():
        print(f"Dang huan luyen: {name} ...")
        # Huấn luyện
        model.fit(X_tr, y_tr)
        
        # Suy luận trên EVAL
        probas = model.predict_proba(X_ev)[:, 1]
        
        # Tính EER & AUC
        eer, roc_auc = compute_eer(y_ev, probas)
        
        print(f"    -> XONG! EER: {eer:.2f}% | AUC: {roc_auc:.2f}%\n")
        
        results.append({
            "Mô hình": name,
            "EER (%)": round(eer, 2),
            "ROC-AUC (%)": round(roc_auc, 2)
        })
    
    # Sắp xếp kết quả theo EER (thấp nhất là tốt nhất)
    df_results = pd.DataFrame(results).sort_values(by="EER (%)", ascending=True)
    
    print("="*60)
    print("BANG XEP HANG CHUNG CUOC (TAP EVAL 71.237 MAU)")
    print("="*60)
    print(df_results.to_markdown(index=False))
    print("="*60)
    
    # Lưu ra file CSV
    out_csv = PROJECT_ROOT / "benchmark_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"[+] Da luu ket qua ra file: {out_csv.name}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
