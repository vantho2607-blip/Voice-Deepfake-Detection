"""
tests/test_step2_preprocessing.py

Kiểm thử tự động cho BƯỚC 2: Tiền xử lý & Lọc nhiễu âm thanh (Audio Preprocessing)
"""

from pathlib import Path
import sqlite3
import numpy as np
import pytest
import soundfile as sf

from src.preprocessing.audio_preprocessor import AudioPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.db"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "test_output"


@pytest.fixture(scope="session")
def preprocessor():
    return AudioPreprocessor(target_sr=16000, norm_target_peak=0.95)


@pytest.fixture(scope="session")
def real_bonafide_path():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT audio_path FROM samples WHERE label='bonafide' LIMIT 1;")
    path = cursor.fetchone()[0]
    conn.close()
    return Path(path)


@pytest.fixture(scope="session")
def real_spoof_path():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT audio_path FROM samples WHERE label='spoof' AND attack_id='A01' LIMIT 1;")
    path = cursor.fetchone()[0]
    conn.close()
    return Path(path)


class TestStep2AudioLoadingAndResampling:
    def test_load_real_audio(self, preprocessor, real_bonafide_path):
        waveform, sr = preprocessor.load_audio(real_bonafide_path)
        assert isinstance(waveform, np.ndarray)
        assert waveform.ndim == 1
        assert len(waveform) > 0
        assert sr == 16000

    def test_ensure_sample_rate_no_op(self, preprocessor):
        # Khi audio đã là 16000 Hz -> không resample
        dummy = np.zeros(16000, dtype=np.float32)
        out, sr, resampled = preprocessor.ensure_sample_rate(dummy, orig_sr=16000)
        assert sr == 16000
        assert not resampled
        assert len(out) == 16000

    def test_ensure_sample_rate_resampling_behavior(self, preprocessor):
        # Khi audio là 8000 Hz hoặc 22050 Hz -> phải resample về 16000
        t = np.linspace(0, 1.0, 8000, endpoint=False, dtype=np.float32)
        sine_8k = np.sin(2 * np.pi * 440 * t)
        out, sr, resampled = preprocessor.ensure_sample_rate(sine_8k, orig_sr=8000)
        assert sr == 16000
        assert resampled
        assert abs(len(out) - 16000) <= 2


class TestStep2SilenceTrimming:
    def test_silence_trimming_removes_leading_and_trailing(self, preprocessor):
        # Tạo tín hiệu: 0.5s im lặng + 1.0s tiếng nói (sine wave) + 0.5s im lặng
        sr = 16000
        silence_pre = np.zeros(int(0.5 * sr), dtype=np.float32)
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        speech = 0.5 * np.sin(2 * np.pi * 440 * t)
        silence_post = np.zeros(int(0.5 * sr), dtype=np.float32)

        audio_with_silence = np.concatenate([silence_pre, speech, silence_post])
        trimmed, info = preprocessor.trim_silence(audio_with_silence, sr=sr)

        assert info["trimmed"]
        assert len(trimmed) < len(audio_with_silence)
        # Thời lượng sau cắt phải xấp xỉ 1.0s tiếng nói
        assert 0.9 <= len(trimmed) / sr <= 1.1

    def test_silence_trimming_handles_flat_zeros(self, preprocessor):
        flat_zero = np.zeros(16000, dtype=np.float32)
        trimmed, info = preprocessor.trim_silence(flat_zero, sr=16000)
        assert not np.isnan(trimmed).any()
        assert not np.isinf(trimmed).any()


class TestStep2AmplitudeNormalization:
    def test_amplitude_normalization_peak_scaling(self, preprocessor):
        raw = np.array([-0.2, 0.4, -0.5, 0.1], dtype=np.float32)
        normalized, info = preprocessor.normalize_amplitude(raw)
        assert info["normalized"]
        assert np.isclose(np.max(np.abs(normalized)), 0.95, atol=1e-5)

    def test_amplitude_normalization_zero_signal_safe(self, preprocessor):
        zeros = np.zeros(100, dtype=np.float32)
        normalized, info = preprocessor.normalize_amplitude(zeros)
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()
        assert np.max(np.abs(normalized)) == 0.0


class TestStep2SpectralSubtraction:
    def test_spectral_subtraction_execution_and_clean_values(self, preprocessor, real_bonafide_path):
        waveform, sr = preprocessor.load_audio(real_bonafide_path)
        denoised, info = preprocessor.spectral_subtraction(waveform, sr=sr)

        assert info["applied"]
        assert len(denoised) == len(waveform)
        assert not np.isnan(denoised).any(), "Spectral subtraction sinh NaN!"
        assert not np.isinf(denoised).any(), "Spectral subtraction sinh Inf!"
        assert np.max(np.abs(denoised)) > 0.0, "Spectral subtraction sinh tín hiệu rỗng!"


class TestStep2PipelineAndSafety:
    def test_bonafide_full_pipeline(self, preprocessor, real_bonafide_path, tmp_path):
        res = preprocessor.process_file(real_bonafide_path, output_dir=tmp_path)
        assert res["file_id"] == real_bonafide_path.stem
        assert not res["has_nan"]
        assert not res["has_inf"]
        assert Path(res["output_path"]).exists()

        # Đọc lại file output
        data, sr = sf.read(res["output_path"])
        assert sr == 16000
        assert len(data) > 0

    def test_spoof_full_pipeline(self, preprocessor, real_spoof_path, tmp_path):
        res = preprocessor.process_file(real_spoof_path, output_dir=tmp_path)
        assert res["file_id"] == real_spoof_path.stem
        assert not res["has_nan"]
        assert not res["has_inf"]
        assert Path(res["output_path"]).exists()

    def test_input_file_not_overwritten(self, preprocessor, real_bonafide_path, tmp_path):
        mtime_before = real_bonafide_path.stat().st_mtime
        size_before = real_bonafide_path.stat().st_size

        preprocessor.process_file(real_bonafide_path, output_dir=tmp_path)

        mtime_after = real_bonafide_path.stat().st_mtime
        size_after = real_bonafide_path.stat().st_size

        assert mtime_before == mtime_after, "File gốc bị thay đổi timestamp!"
        assert size_before == size_after, "File gốc bị thay đổi kích thước!"

    def test_safety_guard_blocks_raw_dir_overwrite(self, preprocessor):
        dummy_audio = np.zeros(16000, dtype=np.float32)
        raw_dir = (
            PROJECT_ROOT / "data" / "LA" / "LA"
            if (PROJECT_ROOT / "data" / "LA" / "LA").exists()
            else PROJECT_ROOT / "data" / "archive" / "LA" / "LA"
        )
        forbidden_path = Path(raw_dir) / "test_forbidden.wav"

        with pytest.raises(PermissionError, match="NGHIÊM CẤM GHI ĐÈ"):
            preprocessor.save_audio(dummy_audio, 16000, forbidden_path, raw_data_dir=raw_dir)
