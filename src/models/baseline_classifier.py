#!/usr/bin/env python3
"""
src/models/baseline_classifier.py

Mô hình phân loại xác suất cơ bản (Probability Baseline Classifier) cho BƯỚC 3:
Sử dụng Logistic Regression kết hợp StandardScaler trong sklearn Pipeline.
Hỗ trợ:
  - Huấn luyện chuẩn mực (fit scaler + classifier CHỈ trên tập TRAIN).
  - Xuất xác suất tự nhiên P(bonafide), P(spoof).
  - Tối ưu ngưỡng và tính toán Equal Error Rate (EER), ROC-AUC, F1, Confusion Matrix.
  - Lưu và tải checkpoint độc lập bằng joblib.
  - Đảm bảo tính tái lập (reproducibility) với random_state cố định.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Tính Equal Error Rate (EER) và ngưỡng EER tương ứng.
    EER là điểm trên đường cong ROC nơi FAR (False Alarm) = FRR (False Reject / Miss).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Tìm chỉ số tối thiểu khoảng cách |fpr - fnr|
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    return eer, eer_threshold


class BaselineClassifier:
    """
    Bộ phân loại baseline nhị phân (0 = bonafide, 1 = spoof).
    """

    CLASS_MAP = {"bonafide": 0, "spoof": 1}
    INV_CLASS_MAP = {0: "bonafide", 1: "spoof"}

    def __init__(
        self,
        class_weight: str = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
        solver: str = "lbfgs",
        c_param: float = 1.0,
    ):
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.c_param = c_param

        # Sklearn Pipeline tích hợp chuẩn hóa và phân loại
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=self.class_weight,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    solver=self.solver,
                    C=self.c_param,
                ),
            ),
        ])
        self.is_fitted = False
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "BaselineClassifier":
        """
        Huấn luyện scaler và classifier strictly CHỈ trên tập TRAIN.
        """
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        if feature_names is not None:
            self.feature_names = list(feature_names)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán phân bố xác suất [P(bonafide), P(spoof)].
        Đảm bảo 0 <= P <= 1 và P(bonafide) + P(spoof) == 1.
        """
        if not self.is_fitted:
            raise ValueError("Mô hình chưa được fit! Hãy gọi fit() trước.")

        probs = self.pipeline.predict_proba(X)
        return probs.astype(np.float64)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Dự đoán nhãn nhị phân (0 hoặc 1) dựa theo threshold cho P(spoof).
        """
        probs = self.predict_proba(X)
        prob_spoof = probs[:, 1]
        return (prob_spoof >= threshold).astype(int)

    def predict_single(
        self, feature_vector: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Dự đoán xác suất cho một mẫu audio duy nhất.
        """
        X = np.atleast_2d(feature_vector)
        probs = self.predict_proba(X)[0]
        prob_bonafide = float(probs[0])
        prob_spoof = float(probs[1])

        prediction_label = "spoof" if prob_spoof >= threshold else "bonafide"

        return {
            "prob_bonafide": round(prob_bonafide, 4),
            "prob_spoof": round(prob_spoof, 4),
            "prediction": prediction_label,
            "threshold": threshold,
            "confidence": round(max(prob_bonafide, prob_spoof), 4),
        }

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Đánh giá toàn diện các metrics trên tập dữ liệu.
        """
        probs = self.predict_proba(X)
        prob_spoof = probs[:, 1]
        y_pred = (prob_spoof >= threshold).astype(int)

        # Basic metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        # Confusion matrix: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Alarm (Bonafide -> Spoof)
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Reject / Miss (Spoof -> Bonafide)

        # ROC-AUC
        try:
            auc = roc_auc_score(y_true, prob_spoof)
        except Exception:
            auc = 0.5

        # Equal Error Rate
        try:
            eer, eer_threshold = compute_eer(y_true, prob_spoof)
        except Exception:
            eer, eer_threshold = 0.5, 0.5

        return {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "eer": round(float(eer), 4),
            "eer_threshold": round(float(eer_threshold), 4),
            "threshold": threshold,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "far_bonafide_as_spoof": round(float(far), 4),
            "frr_spoof_as_bonafide": round(float(frr), 4),
            "total_samples": int(len(y_true)),
            "bonafide_count": int(np.sum(y_true == 0)),
            "spoof_count": int(np.sum(y_true == 1)),
        }

    def save(self, filepath: str | Path, extra_meta: Optional[Dict[str, Any]] = None) -> Path:
        """
        Lưu mô hình và scaler vào file checkpoint (.joblib).
        """
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "pipeline": self.pipeline,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "extra_meta": extra_meta or {},
        }
        joblib.dump(payload, str(p))
        return p

    @classmethod
    def load(cls, filepath: str | Path) -> "BaselineClassifier":
        """
        Tải mô hình từ file checkpoint.
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint không tồn tại: {p}")

        payload = joblib.load(str(p))
        instance = cls(
            class_weight=payload.get("class_weight", "balanced"),
            random_state=payload.get("random_state", 42),
        )
        instance.pipeline = payload["pipeline"]
        instance.is_fitted = payload.get("is_fitted", True)
        instance.feature_names = payload.get("feature_names", [])
        return instance
