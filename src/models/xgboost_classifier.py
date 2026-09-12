"""
src/models/xgboost_classifier.py

Mô hình phân loại dùng XGBoost thay cho Logistic Regression.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    import warnings
    warnings.warn("Thư viện xgboost chưa được cài đặt.")

from src.models.baseline_classifier import BaselineClassifier

class XGBoostBaselineClassifier(BaselineClassifier):
    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        # XGBoost có class_weight thông qua scale_pos_weight, nhưng ta để tự nhiên hoặc tính sau.
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    eval_metric="logloss"
                ),
            ),
        ])
        self.is_fitted = False
        self.feature_names = []

    @classmethod
    def load(cls, filepath: str | Path) -> "XGBoostBaselineClassifier":
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint không tồn tại: {p}")

        payload = joblib.load(str(p))
        instance = cls(
            random_state=payload.get("random_state", 42),
        )
        instance.pipeline = payload["pipeline"]
        instance.is_fitted = payload.get("is_fitted", True)
        instance.feature_names = payload.get("feature_names", [])
        return instance

    def save(self, filepath: str | Path, extra_meta: Optional[Dict[str, Any]] = None) -> Path:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "pipeline": self.pipeline,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "random_state": self.random_state,
            "extra_meta": extra_meta or {},
        }
        joblib.dump(payload, str(p))
        return p
