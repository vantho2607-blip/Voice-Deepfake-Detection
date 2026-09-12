#!/usr/bin/env python3
"""
tests/test_step4_acoustic.py

Bộ kiểm thử tự động (Unit & Integration Tests) cho BƯỚC 4 — Acoustic Chain of Evidence (ACoE).
Bao gồm:
  1. Test cấu trúc dữ liệu chứng cứ (Evidence Dataclasses & JSON serialization).
  2. Test các thử thách âm học (Gain, Additive Noise, Resampling) độc lập:
     - Tính tất định (reproducible seed)
     - Giữ nguyên độ dài waveform
     - Xử lý biên (audio im lặng, audio cực ngắn)
  3. Test ChallengeRunner & Acoustic Consistency Score:
     - Bounded trong đoạn [0, 1]
     - Tính toán chính xác các độ lệch (delta P, evidence delta)
  4. Test các module trích xuất chứng cứ âm học (Spectral, Temporal, Cepstral, Prosodic):
     - F0 tracking trên sóng âm chuẩn (sine 200 Hz -> F0 ~200 Hz, status=available)
     - F0 tracking trên khoảng im lặng / nhiễu vô định hình (status=unavailable, giá trị None an toàn)
     - Kích thước vector MFCC và Deltas (20 dimensions)
     - Phổ tần: Centroid, Rolloff, Flatness [0, 1], Flux >= 0
  5. Test tích hợp đầu-cuối (End-to-End Analysis):
     - Phân tích waveform trong RAM
     - Phân tích file thực tế từ dataset
     - ĐẢM BẢO TUYỆT ĐỐI: File âm thanh gốc KHÔNG BỊ SỬA ĐỔI HOẶC GHI ĐÈ.
"""

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from src.analysis.acoustic_analyzer import AcousticAnalyzer
from src.analysis.acoustic_challenges import (
    AdditiveNoiseChallenge,
    ChallengeRunner,
    GainChallenge,
    ResamplingChallenge,
)
from src.analysis.evidence import (
    AcousticChainOfEvidence,
    CepstralEvidence,
    ChallengeResult,
    ClassifierEvidence,
    MetadataEvidence,
    ProsodicEvidence,
    SpectralEvidence,
    StabilityEvidence,
    TemporalEvidence,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "baseline_logistic_regression.joblib"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.db"


# =====================================================================
# 1. TEST EVIDENCE DATACLASSES & SCHEMA
# =====================================================================
class TestEvidenceDataclasses:
    def test_evidence_structure_and_json_serialization(self):
        """Kiểm tra việc khởi tạo và serialization của cấu trúc ACoE."""
        ev = AcousticChainOfEvidence(
            file_id="TEST_001",
            classifier=ClassifierEvidence(p_bonafide=0.95, p_spoof=0.05, prediction="bonafide"),
            spectral=SpectralEvidence(
                centroid_mean=2000.0,
                centroid_std=300.0,
                rolloff_mean=2500.0,
                rolloff_std=400.0,
                bandwidth_mean=1400.0,
                bandwidth_std=200.0,
                flatness_mean=0.05,
                flatness_std=0.02,
                flux_mean=0.04,
                flux_std=0.01,
            ),
            temporal=TemporalEvidence(
                rms_mean=0.08,
                rms_std=0.04,
                rms_variation=0.5,
                zcr_mean=0.15,
                zcr_std=0.05,
                silence_ratio=0.1,
                voiced_ratio_est=0.6,
            ),
            cepstral=CepstralEvidence(
                mfcc_mean=[0.1] * 20,
                mfcc_std=[0.2] * 20,
                delta_mean=[0.05] * 20,
                delta_std=[0.02] * 20,
                delta2_mean=[0.01] * 20,
                delta2_std=[0.01] * 20,
            ),
            prosodic=ProsodicEvidence(
                status="available",
                f0_mean=210.5,
                f0_std=25.3,
                f0_range=120.0,
                voiced_ratio=0.75,
                pitch_variation=0.12,
            ),
            challenges=[
                ChallengeResult(
                    name="gain_0.90",
                    parameters={"gain_factor": 0.9},
                    p_spoof=0.06,
                    delta_p_spoof=0.01,
                    evidence_delta=0.001,
                )
            ],
            stability=StabilityEvidence(
                acoustic_consistency_score=0.98,
                challenge_count=1,
                mean_abs_delta_p_spoof=0.01,
                mean_evidence_delta=0.001,
                per_challenge_stability={"gain_0.90": 0.99},
            ),
            metadata=MetadataEvidence(
                sample_rate=16000,
                duration_seconds=2.5,
                processing_status="success",
            ),
        )

        d = ev.to_dict()
        assert isinstance(d, dict)
        assert d["file_id"] == "TEST_001"
        assert d["classifier"]["prediction"] == "bonafide"
        assert d["prosodic"]["status"] == "available"
        assert len(d["cepstral"]["mfcc_mean"]) == 20

        json_str = ev.to_json()
        loaded = json.loads(json_str)
        assert loaded["file_id"] == "TEST_001"
        assert loaded["stability"]["acoustic_consistency_score"] == 0.98

    def test_prosodic_unavailable_behavior(self):
        """Kiểm tra trường hợp prosodic không khả dụng (status=unavailable)."""
        p = ProsodicEvidence(status="unavailable")
        assert p.status == "unavailable"
        assert p.f0_mean is None
        assert p.f0_std is None
        assert p.f0_range is None
        assert p.voiced_ratio is None


# =====================================================================
# 2. TEST ACOUSTIC CHALLENGES
# =====================================================================
class TestAcousticChallenges:
    @pytest.fixture
    def sample_sine_wave(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        # Sine wave 200 Hz biên độ 0.5
        wave = 0.5 * np.sin(2 * np.pi * 200.0 * t)
        return wave, sr

    def test_gain_challenge(self, sample_sine_wave):
        wave, sr = sample_sine_wave
        orig_rms = np.sqrt(np.mean(wave**2))

        ch_atten = GainChallenge(gain_factor=0.90)
        p_atten, params_atten = ch_atten.apply(wave, sr)
        atten_rms = np.sqrt(np.mean(p_atten**2))

        assert len(p_atten) == len(wave)
        assert atten_rms < orig_rms
        assert pytest.approx(atten_rms, rel=1e-3) == orig_rms * 0.90
        assert params_atten["gain_factor"] == 0.90
        assert params_atten["gain_db"] < 0

        # Kiểm tra clipping [-1.0, 1.0]
        big_wave = np.ones(100, dtype=np.float32) * 0.95
        ch_boost = GainChallenge(gain_factor=1.5)
        p_boost, _ = ch_boost.apply(big_wave, sr)
        assert np.max(p_boost) <= 1.0

    def test_additive_noise_challenge_reproducibility(self, sample_sine_wave):
        wave, sr = sample_sine_wave

        # 2 lần chạy cùng seed phải cho kết quả giống hệt 100%
        ch1 = AdditiveNoiseChallenge(snr_db=35.0, seed=123)
        ch2 = AdditiveNoiseChallenge(snr_db=35.0, seed=123)
        p1, _ = ch1.apply(wave, sr)
        p2, _ = ch2.apply(wave, sr)
        np.testing.assert_array_equal(p1, p2)

        # Seed khác nhau phải khác nhau
        ch3 = AdditiveNoiseChallenge(snr_db=35.0, seed=456)
        p3, _ = ch3.apply(wave, sr)
        assert not np.array_equal(p1, p3)

    def test_additive_noise_on_silence(self):
        """Kiểm tra việc thêm nhiễu vào tín hiệu im lặng hoàn toàn không gây crash."""
        sr = 16000
        silent_wave = np.zeros(1600, dtype=np.float32)
        ch = AdditiveNoiseChallenge(snr_db=35.0, seed=42)
        perturbed, params = ch.apply(silent_wave, sr)
        assert len(perturbed) == len(silent_wave)
        assert np.all(np.isfinite(perturbed))

    def test_resampling_challenge(self, sample_sine_wave):
        wave, sr = sample_sine_wave
        orig_len = len(wave)

        ch = ResamplingChallenge(intermediate_sr=15200, seed=42)
        perturbed, params = ch.apply(wave, sr)

        # Độ dài sau khi down-up sampling phải giữ nguyên chính xác
        assert len(perturbed) == orig_len
        assert np.all(np.isfinite(perturbed))
        assert params["intermediate_sr"] == 15200

    def test_resampling_on_empty(self):
        ch = ResamplingChallenge(intermediate_sr=15200)
        empty = np.array([], dtype=np.float32)
        res, _ = ch.apply(empty, 16000)
        assert len(res) == 0


# =====================================================================
# 3. TEST CHALLENGE RUNNER & STABILITY
# =====================================================================
class TestChallengeRunner:
    def test_runner_execution_and_score_bounds(self):
        """Kiểm tra ChallengeRunner tính toán điểm trong đoạn [0, 1]."""
        if not MODEL_PATH.exists():
            pytest.skip(f"Không tìm thấy model checkpoint tại {MODEL_PATH}")

        analyzer = AcousticAnalyzer(model_path=MODEL_PATH)
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        wave = (0.3 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)

        feat_vec = analyzer.feature_extractor.extract(wave)
        prob = analyzer.classifier.predict_single(feat_vec)["prob_spoof"]

        runner = ChallengeRunner(seed=42)
        results, stability = runner.run(
            waveform=wave,
            sr=sr,
            classifier=analyzer.classifier,
            feature_extractor=analyzer.feature_extractor,
            original_p_spoof=prob,
            original_feature_vector=feat_vec,
        )

        assert len(results) == 4
        assert 0.0 <= stability.acoustic_consistency_score <= 1.0
        assert stability.mean_abs_delta_p_spoof >= 0.0
        assert stability.mean_evidence_delta >= 0.0
        for r in results:
            assert 0.0 <= r.p_spoof <= 1.0
            assert np.isfinite(r.delta_p_spoof)
            assert r.evidence_delta >= 0.0


# =====================================================================
# 4. TEST EVIDENCE EXTRACTION MODULES
# =====================================================================
class TestEvidenceExtractionModules:
    @pytest.fixture
    def analyzer(self):
        if not MODEL_PATH.exists():
            pytest.skip(f"Không tìm thấy model checkpoint tại {MODEL_PATH}")
        return AcousticAnalyzer(model_path=MODEL_PATH)

    def test_prosodic_pitch_tracking_on_clean_tone(self, analyzer):
        """Kiểm tra F0 tracker phát hiện chính xác tần số cơ bản ~200 Hz."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        tone_200hz = 0.5 * np.sin(2 * np.pi * 200.0 * t)

        prosodic = analyzer._extract_prosodic_evidence(tone_200hz, sr)
        assert prosodic.status == "available"
        assert prosodic.f0_mean is not None
        # Sai số F0 cho phép trong vòng +/- 15 Hz
        assert 185.0 <= prosodic.f0_mean <= 215.0
        assert prosodic.voiced_ratio > 0.5

    def test_prosodic_on_silence_returns_unavailable(self, analyzer):
        """Kiểm tra tín hiệu im lặng phải trả về status='unavailable' và None an toàn."""
        sr = 16000
        silence = np.zeros(16000, dtype=np.float32)

        prosodic = analyzer._extract_prosodic_evidence(silence, sr)
        assert prosodic.status == "unavailable"
        assert prosodic.f0_mean is None
        assert prosodic.f0_std is None
        assert prosodic.f0_range is None
        assert prosodic.voiced_ratio is None

    def test_spectral_evidence_properties(self, analyzer):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        wave = (0.4 * np.sin(2 * np.pi * 500 * t) + 0.2 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)

        from scipy import signal
        freqs, _, Zxx = signal.stft(wave, fs=sr, nperseg=analyzer.n_fft, noverlap=analyzer.n_fft - analyzer.hop_length)
        mag = np.abs(Zxx)
        pwr = mag**2

        spectral = analyzer._extract_spectral_evidence(mag, pwr, freqs)
        assert spectral.centroid_mean > 0.0
        assert spectral.rolloff_mean >= spectral.centroid_mean * 0.5
        assert 0.0 <= spectral.flatness_mean <= 1.0
        assert spectral.flux_mean >= 0.0

    def test_temporal_evidence_properties(self, analyzer):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        wave = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        temporal = analyzer._extract_temporal_evidence(wave, sr)
        assert temporal.rms_mean > 0.0
        assert 0.0 <= temporal.zcr_mean <= 1.0
        assert 0.0 <= temporal.silence_ratio <= 1.0
        assert 0.0 <= temporal.voiced_ratio_est <= 1.0

    def test_cepstral_evidence_dimensions(self, analyzer):
        sr = 16000
        wave = np.random.RandomState(42).randn(16000).astype(np.float32) * 0.1

        cepstral = analyzer._extract_cepstral_evidence(wave)
        assert len(cepstral.mfcc_mean) == 20
        assert len(cepstral.mfcc_std) == 20
        assert len(cepstral.delta_mean) == 20
        assert len(cepstral.delta_std) == 20
        assert len(cepstral.delta2_mean) == 20
        assert len(cepstral.delta2_std) == 20


# =====================================================================
# 5. TEST END-TO-END & SAFETY GUARD
# =====================================================================
class TestEndToEndAndSafety:
    @pytest.fixture
    def analyzer(self):
        if not MODEL_PATH.exists():
            pytest.skip(f"Không tìm thấy model checkpoint tại {MODEL_PATH}")
        return AcousticAnalyzer(model_path=MODEL_PATH)

    def test_analyze_waveform_in_ram(self, analyzer):
        """Phân tích một waveform hoàn chỉnh trong RAM."""
        sr = 16000
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False, dtype=np.float32)
        wave = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

        evidence = analyzer.analyze_waveform(wave, sr=sr, file_id="ram_test")
        assert evidence.file_id == "ram_test"
        assert evidence.metadata.processing_status == "success"
        assert 0.0 <= evidence.classifier.p_spoof <= 1.0
        assert 0.0 <= evidence.stability.acoustic_consistency_score <= 1.0
        assert len(evidence.challenges) == 4

    def test_real_audio_file_unmodified_guard(self, analyzer):
        """
        Kiểm tra phân tích trên file thật từ dataset ASVspoof 2019:
        ĐẢM BẢO TUYỆT ĐỐI: File gốc không bị ghi đè, hash file và mtime hoàn toàn không đổi.
        """
        if not DB_PATH.exists():
            pytest.skip(f"Không tìm thấy database tại {DB_PATH}")

        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT audio_path FROM samples WHERE split = 'dev' LIMIT 1").fetchone()
        conn.close()

        if not row:
            pytest.skip("Không tìm thấy mẫu nào trong DB.")

        target_file = Path(row[0])
        assert target_file.exists(), f"File âm thanh không tồn tại: {target_file}"

        # 1. Đo hash SHA256 và kích thước ban đầu của file
        def compute_sha256(path: Path) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()

        orig_hash = compute_sha256(target_file)
        orig_size = target_file.stat().st_size
        orig_mtime = target_file.stat().st_mtime

        # 2. Thực hiện phân tích ACoE
        evidence = analyzer.analyze_file(target_file, run_challenges=True)

        # 3. Kiểm tra kết quả
        assert evidence.file_id == target_file.stem
        assert evidence.metadata.processing_status == "success"
        assert evidence.metadata.duration_seconds > 0.0
        assert evidence.classifier.prediction in ("bonafide", "spoof")
        assert 0.0 <= evidence.stability.acoustic_consistency_score <= 1.0

        # 4. KIỂM CHỨNG AN TOÀN TUYỆT ĐỐI
        after_hash = compute_sha256(target_file)
        after_size = target_file.stat().st_size
        after_mtime = target_file.stat().st_mtime

        assert orig_hash == after_hash, "NGUY HIỂM: Hash của file gốc đã bị thay đổi!"
        assert orig_size == after_size, "NGUY HIỂM: Kích thước của file gốc đã bị thay đổi!"
        assert orig_mtime == after_mtime, "NGUY HIỂM: Modification time của file gốc đã bị thay đổi!"
