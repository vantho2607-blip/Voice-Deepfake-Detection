import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_cache(prefix: str, split: str, n_samples: int, seed: int = 42):
    cache_path = PROJECT_ROOT / "data" / "processed" / "features_cache" / f"{prefix}_{split}_{n_samples}_s{seed}.npz"
    data = np.load(cache_path, allow_pickle=True)
    return data['X'], data['y'], data['meta']

def main():
    # 1. Load & Train
    X_tr_lfcc, y_tr, _ = load_cache("base", "train", 25380)
    X_ev_lfcc, y_ev, _ = load_cache("base", "eval", 71237)
    X_tr_xlsr, _, _ = load_cache("xlsr_base", "train", 25380)
    X_ev_xlsr, _, _ = load_cache("xlsr_base", "eval", 71237)
    
    X_tr = np.hstack([X_tr_lfcc, X_tr_xlsr])
    X_ev = np.hstack([X_ev_lfcc, X_ev_xlsr])
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ev = scaler.transform(X_ev)
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_tr, y_tr)
    
    # 2. Predict with optimal threshold (~0.49 calculated previously, we use 0.5 for simplicity in confusion matrix)
    probas = clf.predict_proba(X_ev)[:, 1]
    threshold = 0.5
    preds = (probas >= threshold).astype(int)
    
    # 3. Calculate metrics
    cm = confusion_matrix(y_ev, preds)
    # y_ev: 0 = Bonafide (Thật), 1 = Spoof (Giả)
    # cm[0,0] = True Negative (Thật đoán Thật)
    # cm[0,1] = False Positive (Thật đoán Giả)
    # cm[1,0] = False Negative (Giả đoán Thật)
    # cm[1,1] = True Positive (Giả đoán Giả)
    
    tn, fp, fn, tp = cm.ravel()
    
    # 4. Generate Intuitive Plots
    plt.rcParams['font.sans-serif'] = ['Arial', 'Segoe UI']
    fig = plt.figure(figsize=(16, 7))
    
    # --- PLOT 1: CONFUSION MATRIX ---
    ax1 = plt.subplot(1, 2, 1)
    
    # Đổi label cho dễ hiểu
    labels = ['Giọng THẬT\n(Bona fide)', 'Giọng GIẢ MẠO\n(Spoof)']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=labels, yticklabels=labels, ax=ax1,
                annot_kws={"size": 18, "weight": "bold"})
    
    ax1.set_title('Ma Trận Nhầm Lẫn (Khám bệnh cho Mô hình)\nTrên 71.000 tệp EVAL', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('THỰC TẾ (Đáp án đúng)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('MÁY TÍNH DỰ ĐOÁN', fontsize=14, fontweight='bold')
    
    # --- PLOT 2: PIE CHARTS (TỶ LỆ BẮT TRÚNG) ---
    ax2 = plt.subplot(1, 4, 3)
    ax3 = plt.subplot(1, 4, 4)
    
    # Pie chart cho Giọng Thật
    real_total = tn + fp
    real_sizes = [tn, fp]
    real_labels = ['Chính xác\n(Máy bảo Thật)', 'Báo động nhầm\n(Máy bảo Giả)']
    real_colors = ['#2ecc71', '#e74c3c']
    
    ax2.pie(real_sizes, labels=real_labels, colors=real_colors, autopct='%1.1f%%', 
            startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    ax2.set_title(f'Khả năng nhận diện\nGIỌNG THẬT\n(Tổng: {real_total} tệp)', fontsize=14, fontweight='bold')
    
    # Pie chart cho Giọng Giả
    fake_total = fn + tp
    fake_sizes = [tp, fn]
    fake_labels = ['Bắt trúng\n(Máy bảo Giả)', 'Bỏ lọt\n(Máy bảo Thật)']
    fake_colors = ['#e74c3c', '#95a5a6']
    
    ax3.pie(fake_sizes, labels=fake_labels, colors=fake_colors, autopct='%1.1f%%', 
            startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    ax3.set_title(f'Khả năng nhận diện\nGIỌNG GIẢ MẠO\n(Tổng: {fake_total} tệp)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    out_path = PROJECT_ROOT / "truc_quan_ket_qua.png"
    plt.savefig(out_path, dpi=300)

if __name__ == "__main__":
    main()
