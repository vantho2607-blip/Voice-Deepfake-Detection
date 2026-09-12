import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_cache(prefix: str, split: str, n_samples: int, seed: int = 42):
    cache_path = PROJECT_ROOT / "data" / "processed" / "features_cache" / f"{prefix}_{split}_{n_samples}_s{seed}.npz"
    data = np.load(cache_path, allow_pickle=True)
    return data['X'], data['y'], data['meta']

def main():
    print("[+] Loading Data...")
    # Load LFCC
    X_tr_lfcc, y_tr, _ = load_cache("base", "train", 25380)
    X_ev_lfcc, y_ev, _ = load_cache("base", "eval", 71237)
    
    # Load XLS-R
    X_tr_xlsr, _, _ = load_cache("xlsr_base", "train", 25380)
    X_ev_xlsr, _, _ = load_cache("xlsr_base", "eval", 71237)
    
    # Early Fusion
    X_tr = np.hstack([X_tr_lfcc, X_tr_xlsr])
    X_ev = np.hstack([X_ev_lfcc, X_ev_xlsr])
    
    print("[+] Scaling...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ev = scaler.transform(X_ev)
    
    print("[+] Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_tr, y_tr)
    
    print("[+] Predicting on EVAL set...")
    # Lấy xác suất là Spoof (Class 1)
    probas = clf.predict_proba(X_ev)[:, 1]
    
    print("[+] Generating Plot...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # -------------------------------------------------------------
    # PLOT 1: PHÂN BỐ XÁC SUẤT (SCORE DISTRIBUTION)
    # -------------------------------------------------------------
    bonafide_scores = probas[y_ev == 0]
    spoof_scores = probas[y_ev == 1]
    
    sns.histplot(bonafide_scores, bins=50, color='blue', alpha=0.6, label='Bona fide (Thật)', ax=axes[0], stat='density', kde=True)
    sns.histplot(spoof_scores, bins=50, color='red', alpha=0.6, label='Spoof (Giả mạo)', ax=axes[0], stat='density', kde=True)
    
    # Vẽ đường Threshold (mặc định 0.5)
    axes[0].axvline(0.5, color='black', linestyle='--', label='Decision Boundary (0.5)')
    axes[0].set_title('Phân bố xác suất của Logistic Regression (Tập EVAL)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Xác suất được dự đoán là Giả Mạo (Spoof Probability)', fontsize=12)
    axes[0].set_ylabel('Mật độ (Density)', fontsize=12)
    axes[0].legend()
    
    # -------------------------------------------------------------
    # PLOT 2: PCA 2D SCATTER PLOT CỦA 1152-d FEATURES
    # -------------------------------------------------------------
    print("[+] Running PCA for visualization...")
    # Lấy ngẫu nhiên 5000 mẫu để vẽ cho nhẹ (1000 thật, 4000 giả)
    idx_bonafide = np.where(y_ev == 0)[0]
    idx_spoof = np.where(y_ev == 1)[0]
    
    np.random.seed(42)
    idx_bonafide_sub = np.random.choice(idx_bonafide, min(1000, len(idx_bonafide)), replace=False)
    idx_spoof_sub = np.random.choice(idx_spoof, min(4000, len(idx_spoof)), replace=False)
    
    idx_sub = np.concatenate([idx_bonafide_sub, idx_spoof_sub])
    X_sub = X_ev[idx_sub]
    y_sub = y_ev[idx_sub]
    probas_sub = probas[idx_sub]
    
    # PCA giảm từ 1152 chiều xuống 2 chiều
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sub)
    
    # Vẽ Scatter
    scatter = axes[1].scatter(X_pca[y_sub==1, 0], X_pca[y_sub==1, 1], c=probas_sub[y_sub==1], cmap='Reds', alpha=0.5, label='Spoof', s=15)
    axes[1].scatter(X_pca[y_sub==0, 0], X_pca[y_sub==0, 1], c='blue', alpha=0.7, label='Bona fide', s=15, edgecolors='k')
    
    axes[1].set_title('Không gian đặc trưng 1152-d chiếu xuống 2D (PCA)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Thành phần chính 1 (Principal Component 1)')
    axes[1].set_ylabel('Thành phần chính 2 (Principal Component 2)')
    
    # Tạo colorbar cho Spoof
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Xác suất do Logistic Regression phán xét', rotation=270, labelpad=15)
    
    axes[1].legend()
    
    plt.tight_layout()
    
    out_path = PROJECT_ROOT / "eval_decision_boundary.png"
    plt.savefig(out_path, dpi=300)
    print(f"[+] Đã lưu biểu đồ thành công tại: {out_path}")

if __name__ == "__main__":
    main()
