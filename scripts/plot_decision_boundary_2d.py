import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_cache(prefix: str, split: str, n_samples: int, seed: int = 42):
    cache_path = PROJECT_ROOT / "data" / "processed" / "features_cache" / f"{prefix}_{split}_{n_samples}_s{seed}.npz"
    data = np.load(cache_path, allow_pickle=True)
    return data['X'], data['y'], data['meta']

def main():
    print("[+] Loading Data...")
    X_ev_lfcc, y_ev, _ = load_cache("base", "eval", 71237)
    X_ev_xlsr, _, _ = load_cache("xlsr_base", "eval", 71237)
    
    # 1. Ghép 1152 chiều và chuẩn hóa
    X_ev = np.hstack([X_ev_lfcc, X_ev_xlsr])
    scaler = StandardScaler()
    X_ev_scaled = scaler.fit_transform(X_ev)
    
    # 2. Rút gọn không gian 1152 chiều xuống còn đúng 2 mặt phẳng (PCA)
    print("[+] Running PCA to 2D...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_ev_scaled)
    
    # Lấy mẫu ngẫu nhiên 3000 điểm để vẽ cho đỡ rối mắt
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_pca), 3000, replace=False)
    X_plot = X_pca[sample_indices]
    y_plot = y_ev[sample_indices]
    
    # 3. Huấn luyện một mô hình Logistic Regression MINH HỌA trực tiếp trên 2D
    print("[+] Fitting 2D Logistic Regression for Decision Boundary...")
    clf_2d = LogisticRegression(class_weight='balanced')
    clf_2d.fit(X_plot, y_plot)
    
    # 4. Vẽ "Đường thẳng đặc trưng" (Decision Boundary)
    plt.rcParams['font.sans-serif'] = ['Arial', 'Segoe UI']
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Vẽ nền (Khu vực màu xanh và khu vực màu đỏ bị chia cắt bởi đường thẳng)
    disp = DecisionBoundaryDisplay.from_estimator(
        clf_2d, X_plot, response_method="predict",
        cmap="coolwarm", alpha=0.3, ax=ax,
        xlabel="Thành phần không gian 1 (PCA 1)",
        ylabel="Thành phần không gian 2 (PCA 2)"
    )
    
    # Vẽ các dấu chấm
    # Giọng thật (0) = Xanh dương
    scatter = ax.scatter(X_plot[y_plot == 0, 0], X_plot[y_plot == 0, 1], 
                         c='blue', edgecolor='k', label='Giọng Thật (Bona fide)', s=30, alpha=0.8)
    
    # Giọng giả (1) = Đỏ
    ax.scatter(X_plot[y_plot == 1, 0], X_plot[y_plot == 1, 1], 
               c='red', edgecolor='k', label='Giọng Giả (Spoof)', s=30, alpha=0.6, marker='x')
    
    # Vẽ ĐƯỜNG THẲNG ĐẶC TRƯNG (Decision boundary line) manually để làm nổi bật
    w = clf_2d.coef_[0]
    b = clf_2d.intercept_[0]
    # w0*x + w1*y + b = 0 => y = -(w0*x + b)/w1
    x_line = np.linspace(X_plot[:, 0].min(), X_plot[:, 0].max(), 100)
    y_line = -(w[0] * x_line + b) / w[1]
    ax.plot(x_line, y_line, color='black', linewidth=3, linestyle='--', label='Đường ranh giới tuyến tính (Logistic Regression)')
    
    # Giới hạn trục tọa độ
    ax.set_ylim(X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1)
    
    ax.set_title("ĐƯỜNG THẲNG PHÂN ĐỊNH (DECISION BOUNDARY)\nCỦA LOGISTIC REGRESSION (Minh họa trên không gian 2D)", 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='best', fontsize=12)
    
    out_path = PROJECT_ROOT / "duong_thang_logistic_regression.png"
    plt.savefig(out_path, dpi=300)
    print(f"[+] Done! Saved to {out_path}")

if __name__ == "__main__":
    main()
