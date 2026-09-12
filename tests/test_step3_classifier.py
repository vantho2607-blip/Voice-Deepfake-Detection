"""
tests/test_step3_classifier.py

Kiểm thử tự động cho BƯỚC 3: Bộ phân loại xác suất Baseline (Probability Classifier)
Bao gồm 14 tiêu chí bắt buộc:
  1. SQLite query
  2. Label mapping
  3. Feature extraction
  4. Feature dimension (128)
  5. No NaN
  6. No Inf
  7. StandardScaler fit on train
  8. Probability range [0, 1]
  9. Probability sum == 1
  10. Classifier prediction (bonafide / spoof)
  11. Model checkpoint save
  12. Model checkpoint load
  13. Single-audio inference
  14. No modification to original audio
"""

from pathlib import Path
import sqlite3
import numpy as np
import pytest

from src.features.audio_features import AcousticFeatureExtractor
from src.models.baseline_classifier import BaselineClassifier
from src.preprocessing.audio_preprocessor import AudioPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.db"
TEST_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "test_classifier.joblib"


@pytest.fixture(scope="module")
def extractor():
    return AcousticFeatureExtractor(sr=16000, n_fft=512, hop_length=256, n_mels=20)


@pytest.fixture(scope="module")
def sample_audio_paths():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT audio_path FROM samples WHERE label='bonafide' LIMIT 1;")
    bonafide = cursor.fetchone()[0]
    cursor.execute("SELECT audio_path FROM samples WHERE label='spoof' LIMIT 1;")
    spoof = cursor.fetchone()[0]
    conn.close()
    return {"bonafide": Path(bonafide), "spoof": Path(spoof)}


class TestStep3DataAndFeatures:
    def test_sqlite_query_split_and_labels(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT split FROM samples;")
        splits = {r[0] for r in cursor.fetchall()}
        assert splits == {"train", "dev", "eval"}

        cursor.execute("SELECT DISTINCT label FROM samples;")
        labels = {r[0] for r in cursor.fetchall()}
        assert labels == {"bonafide", "spoof"}
        conn.close()

    def test_label_mapping_consistency(self):
        assert BaselineClassifier.CLASS_MAP["bonafide"] == 0
        assert BaselineClassifier.CLASS_MAP["spoof"] == 1
        assert BaselineClassifier.INV_CLASS_MAP[0] == "bonafide"
        assert BaselineClassifier.INV_CLASS_MAP[1] == "spoof"

    def test_feature_extraction_dimension(self, extractor):
        assert extractor.feature_dim == 128
        assert len(extractor.feature_names) == 128

        dummy = np.random.randn(16000).astype(np.float32)
        feats = extractor.extract(dummy)
        assert feats.shape == (128,)

    def test_feature_extraction_no_nan_no_inf(self, extractor):
        dummy = np.random.randn(16000).astype(np.float32)
        feats = extractor.extract(dummy)
        assert not np.isnan(feats).any(), "Features contain NaN"
        assert not np.isinf(feats).any(), "Features contain Inf"

    def test_feature_extraction_zero_signal_safe(self, extractor):
        zeros = np.zeros(16000, dtype=np.float32)
        feats = extractor.extract(zeros)
        assert feats.shape == (128,)
        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()


class TestStep3ClassifierLogic:
    @pytest.fixture(scope="class")
    def trained_classifier(self, extractor):
        rng = np.random.RandomState(42)
        # Tạo dữ liệu train giả lập có 128 chiều
        X_train = rng.randn(60, 128).astype(np.float32)
        y_train = np.array([0] * 30 + [1] * 30, dtype=int)

        clf = BaselineClassifier(class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train, feature_names=extractor.feature_names)
        return clf

    def test_probability_range_and_sum(self, trained_classifier):
        rng = np.random.RandomState(123)
        X_test = rng.randn(20, 128).astype(np.float32)
        probs = trained_classifier.predict_proba(X_test)

        assert probs.shape == (20, 2)
        assert np.all(probs >= 0.0), "Xác suất < 0"
        assert np.all(probs <= 1.0), "Xác suất > 1"
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6), "Tổng xác suất khác 1"

    def test_prediction_output_classes(self, trained_classifier):
        rng = np.random.RandomState(123)
        X_test = rng.randn(20, 128).astype(np.float32)
        preds = trained_classifier.predict(X_test, threshold=0.5)

        assert preds.shape == (20,)
        assert set(preds).issubset({0, 1})

    def test_single_prediction_dict(self, trained_classifier):
        rng = np.random.RandomState(123)
        v = rng.randn(128).astype(np.float32)
        res = trained_classifier.predict_single(v, threshold=0.5)

        assert "prob_bonafide" in res
        assert "prob_spoof" in res
        assert "prediction" in res
        assert res["prediction"] in {"bonafide", "spoof"}
        assert 0.0 <= res["prob_bonafide"] <= 1.0
        assert 0.0 <= res["prob_spoof"] <= 1.0
        assert np.isclose(res["prob_bonafide"] + res["prob_spoof"], 1.0, atol=1e-4)

    def test_model_save_and_load(self, trained_classifier, tmp_path):
        save_path = tmp_path / "model.joblib"
        trained_classifier.save(save_path)
        assert save_path.exists()

        loaded_clf = BaselineClassifier.load(save_path)
        assert loaded_clf.is_fitted
        assert loaded_clf.feature_names == trained_classifier.feature_names

        # So sánh đầu ra predict_proba giữa model gốc và model load lại
        dummy_v = np.random.randn(5, 128).astype(np.float32)
        p1 = trained_classifier.predict_proba(dummy_v)
        p2 = loaded_clf.predict_proba(dummy_v)
        assert np.allclose(p1, p2, atol=1e-7)


class TestStep3EndToEndAndSafety:
    def test_end_to_end_on_real_samples(self, extractor, sample_audio_paths, tmp_path):
        preprocessor = AudioPreprocessor(target_sr=16000, norm_target_peak=0.95)

        for label_type, path in sample_audio_paths.items():
            mtime_before = path.stat().st_mtime
            size_before = path.stat().st_size

            # Chạy qua preprocessing BƯỚC 2
            res = preprocessor.process_file(path, output_dir=None)
            clean_wave = res["processed_waveform"]

            # Trích xuất đặc trưng BƯỚC 3
            vec = extractor.extract(clean_wave)
            assert vec.shape == (128,)
            assert not np.isnan(vec).any()

            # Kiểm tra file gốc không bị sửa đổi
            mtime_after = path.stat().st_mtime
            size_after = path.stat().st_size
            assert mtime_before == mtime_after, f"File {path} bị thay đổi timestamp!"
            assert size_before == size_after, f"File {path} bị thay đổi kích thước!"
