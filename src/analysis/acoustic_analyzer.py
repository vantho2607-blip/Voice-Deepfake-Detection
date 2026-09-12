#!/usr/bin/env python3
"""
src/analysis/acoustic_analyzer.py

Lớp phân tích âm học chính (Acoustic Analyzer) cho BƯỚC 4 — Acoustic Chain of Evidence (ACoE):
Tích hợp:
  1. Tải mô hình phân loại Step 3 (BaselineClassifier checkpoint).
  2. Tiền xử lý âm thanh Step 2 (AudioPreprocessor).
  3. Trích xuất đa chiều chứng cứ âm học:
     - Spectral Evidence (Centroid, Rolloff, Bandwidth, Flatness, Flux)
     - Temporal Evidence (RMS, ZCR, Silence ratio, Voiced ratio)
     - Cepstral Evidence (MFCC, Delta, Delta-Delta)
     - Prosodic Evidence (F0 fundamental frequency, range, variation hoặc unavailable)
  4. Thực thi Acoustic Challenges (Gain, Additive Noise, Resampling).
  5. Tính điểm ổn định & nhất quán âm học (Acoustic Consistency Score).
  6. Xuất cấu trúc bản ghi ACoE hoàn chỉnh.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from src.features.audio_features import AcousticFeatureExtractor
from src.models.baseline_classifier import BaselineClassifier
from src.preprocessing.audio_preprocessor import AudioPreprocessor

from .acoustic_challenges import ChallengeRunner
from .evidence import (
    AcousticChainOfEvidence,
    CepstralEvidence,
    ClassifierEvidence,
    MetadataEvidence,
    ProsodicEvidence,
    SpectralEvidence,
    TemporalEvidence,
)


class AcousticAnalyzer:
    """
    Bộ phân tích chuỗi chứng cứ âm học ACoE.
    """

    def __init__(
        self,
        model_path: str | Path,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 20,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint mô hình Step 3 tại: {model_path}")

        # Tải mô hình Step 3 và các components
        self.classifier = BaselineClassifier.load(model_path)
        self.preprocessor = AudioPreprocessor(target_sr=sr, norm_target_peak=0.95)
        self.feature_extractor = AcousticFeatureExtractor(
            sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

    # -------------------------------------------------------------
    # CÁC HÀM TRÍCH XUẤT CHỨNG CỨ CHI TIẾT
    # -------------------------------------------------------------
    def _extract_spectral_evidence(
        self, magnitude: np.ndarray, power: np.ndarray, freqs: np.ndarray
    ) -> SpectralEvidence:
        """Trích xuất chứng cứ phổ tần (Spectral Evidence)."""
        eps = 1e-12
        freq_weights = freqs[:, None]
        total_mag = np.sum(magnitude, axis=0) + eps

        # 1. Centroid
        centroid = np.sum(freq_weights * magnitude, axis=0) / total_mag

        # 2. Bandwidth
        deviation_sq = (freq_weights - centroid) ** 2
        bandwidth = np.sqrt(np.sum(deviation_sq * magnitude, axis=0) / total_mag)

        # 3. Rolloff (85% năng lượng tích lũy)
        cum_power = np.cumsum(power, axis=0)
        rolloff_idx = np.apply_along_axis(
            lambda col: np.searchsorted(col, 0.85 * col[-1]), 0, cum_power
        )
        rolloff = freqs[rolloff_idx]

        # 4. Flatness (Tỷ số giữa trung bình hình học và trung bình số học)
        log_p = np.log(power + eps)
        geom_mean = np.exp(np.mean(log_p, axis=0))
        arith_mean = np.mean(power, axis=0) + eps
        flatness = geom_mean / arith_mean
        flatness = np.clip(flatness, 0.0, 1.0)

        # 5. Flux (Độ biến thiên phổ giữa 2 frame liên tiếp)
        if magnitude.shape[1] > 1:
            flux = np.sqrt(np.sum(np.diff(magnitude, axis=1) ** 2, axis=0))
        else:
            flux = np.array([0.0], dtype=np.float32)

        return SpectralEvidence(
            centroid_mean=round(float(np.mean(centroid)), 2),
            centroid_std=round(float(np.std(centroid)), 2),
            rolloff_mean=round(float(np.mean(rolloff)), 2),
            rolloff_std=round(float(np.std(rolloff)), 2),
            bandwidth_mean=round(float(np.mean(bandwidth)), 2),
            bandwidth_std=round(float(np.std(bandwidth)), 2),
            flatness_mean=round(float(np.mean(flatness)), 4),
            flatness_std=round(float(np.std(flatness)), 4),
            flux_mean=round(float(np.mean(flux)), 4),
            flux_std=round(float(np.std(flux)), 4),
        )

    def _extract_temporal_evidence(self, waveform: np.ndarray, sr: int) -> TemporalEvidence:
        """Trích xuất chứng cứ thời gian (Temporal Evidence)."""
        num_samples = len(waveform)
        if num_samples == 0:
            return TemporalEvidence(
                rms_mean=0.0,
                rms_std=0.0,
                rms_variation=0.0,
                zcr_mean=0.0,
                zcr_std=0.0,
                silence_ratio=1.0,
                voiced_ratio_est=0.0,
            )

        frame_len = self.n_fft
        hop_len = self.hop_length
        num_frames = max(1, 1 + (num_samples - frame_len) // hop_len)

        rms_list = []
        zcr_list = []
        low_energy_frames = 0

        for i in range(num_frames):
            start = i * hop_len
            end = min(num_samples, start + frame_len)
            chunk = waveform[start:end]
            if len(chunk) < 2:
                continue

            r = float(np.sqrt(np.mean(chunk**2)))
            z = float(np.mean(np.abs(np.diff(np.signbit(chunk)))))
            rms_list.append(r)
            zcr_list.append(z)

            # Ngưỡng im lặng RMS < 0.01 (-40 dBFS)
            if r < 0.01:
                low_energy_frames += 1

        rms_arr = np.array(rms_list, dtype=np.float32) if rms_list else np.array([0.0])
        zcr_arr = np.array(zcr_list, dtype=np.float32) if zcr_list else np.array([0.0])

        r_mean = float(np.mean(rms_arr))
        r_std = float(np.std(rms_arr))
        r_var = r_std / r_mean if r_mean > 1e-6 else 0.0
        silence_ratio = float(low_energy_frames / max(1, len(rms_list)))

        # Ước lượng voiced: frame có năng lượng tốt và ZCR thấp (< 0.2)
        voiced_frames = np.sum((rms_arr >= 0.02) & (zcr_arr < 0.25))
        voiced_ratio_est = float(voiced_frames / max(1, len(rms_list)))

        return TemporalEvidence(
            rms_mean=round(r_mean, 4),
            rms_std=round(r_std, 4),
            rms_variation=round(r_var, 4),
            zcr_mean=round(float(np.mean(zcr_arr)), 4),
            zcr_std=round(float(np.std(zcr_arr)), 4),
            silence_ratio=round(silence_ratio, 4),
            voiced_ratio_est=round(voiced_ratio_est, 4),
        )

    def _extract_cepstral_evidence(self, waveform: np.ndarray) -> CepstralEvidence:
        """Trích xuất chứng cứ cepstral (MFCC, Deltas) dựa trên component Step 3."""
        if len(waveform) < self.n_fft:
            waveform = np.pad(waveform, (0, self.n_fft - len(waveform)))

        # STFT
        _, _, Zxx = signal.stft(
            waveform,
            fs=self.sr,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            window="hann",
            boundary="zeros",
        )
        power = np.abs(Zxx) ** 2

        # Mel + MFCC
        mel_energies = np.dot(self.feature_extractor.fbank, power)
        log_mel = np.log(mel_energies + 1e-10)
        from scipy import fft as sfft
        mfcc = sfft.dct(log_mel, type=2, axis=0, norm="ortho").T  # (T, n_mels)

        delta1 = self.feature_extractor._compute_deltas(mfcc)
        delta2 = self.feature_extractor._compute_deltas(delta1)

        def to_rounded_list(arr, decimals=4):
            return [round(float(v), decimals) for v in arr]

        return CepstralEvidence(
            mfcc_mean=to_rounded_list(np.mean(mfcc, axis=0)),
            mfcc_std=to_rounded_list(np.std(mfcc, axis=0)),
            delta_mean=to_rounded_list(np.mean(delta1, axis=0)),
            delta_std=to_rounded_list(np.std(delta1, axis=0)),
            delta2_mean=to_rounded_list(np.mean(delta2, axis=0)),
            delta2_std=to_rounded_list(np.std(delta2, axis=0)),
        )

    def _extract_prosodic_evidence(self, waveform: np.ndarray, sr: int) -> ProsodicEvidence:
        """
        Trích xuất chứng cứ cao độ / ngữ điệu (Prosodic Evidence).
        Sử dụng thuật toán tự tương quan chuẩn hóa (Normalized Autocorrelation F0 Tracker).
        Nếu audio im lặng, không có tiếng nói hoặc không đáng tin cậy:
          -> Trả về status='unavailable' và các giá trị None (không chế tạo giá trị ảo).
        """
        if len(waveform) == 0:
            return ProsodicEvidence(status="unavailable")

        frame_len = int(0.04 * sr)  # 40 ms = 640 samples
        hop_len = int(0.01 * sr)  # 10 ms = 160 samples
        tau_min = int(sr / 400.0)  # max F0 = 400 Hz (40 samples)
        tau_max = int(sr / 60.0)  # min F0 = 60 Hz (266 samples)

        if len(waveform) < frame_len:
            return ProsodicEvidence(status="unavailable")

        f0_list = []
        total_frames = 0

        for i in range(0, len(waveform) - frame_len, hop_len):
            total_frames += 1
            frame = waveform[i : i + frame_len]
            frame = frame - np.mean(frame)
            norm = np.sum(frame**2)
            if norm < 1e-6:
                continue

            # Tự tương quan (Autocorrelation)
            acorr = np.correlate(frame, frame, mode="full")
            acorr = acorr[len(frame) - 1 :]
            acorr = acorr / (acorr[0] + 1e-12)

            search_region = acorr[tau_min : min(tau_max, len(acorr))]
            if len(search_region) == 0:
                continue

            peak_idx = int(np.argmax(search_region)) + tau_min
            if peak_idx == 0:
                continue
            peak_val = acorr[peak_idx]

            # Ngưỡng voiced tối thiểu 0.35
            if peak_val >= 0.35:
                f0 = sr / peak_idx
                if 60.0 <= f0 <= 400.0:
                    f0_list.append(f0)

        # Cần tối thiểu 5 frame voiced để có thống kê tin cậy
        if len(f0_list) >= 5:
            f0_arr = np.array(f0_list, dtype=np.float32)
            f0_m = float(np.mean(f0_arr))
            f0_s = float(np.std(f0_arr))
            f0_rng = float(np.ptp(f0_arr))
            v_ratio = float(len(f0_list) / max(1, total_frames))
            p_var = f0_s / f0_m if f0_m > 1e-6 else 0.0

            return ProsodicEvidence(
                status="available",
                f0_mean=round(f0_m, 2),
                f0_std=round(f0_s, 2),
                f0_range=round(f0_rng, 2),
                voiced_ratio=round(v_ratio, 4),
                pitch_variation=round(p_var, 4),
            )
        else:
            return ProsodicEvidence(status="unavailable")

    # -------------------------------------------------------------
    # PHÂN TÍCH TỔNG THỂ (ANALYZE FILE & WAVEFORM)
    # -------------------------------------------------------------
    def analyze_waveform(
        self,
        clean_waveform: np.ndarray,
        sr: int = 16000,
        file_id: str = "in_memory",
        run_challenges: bool = True,
        challenge_seed: int = 42,
    ) -> AcousticChainOfEvidence:
        """
        Phân tích chuỗi chứng cứ ACoE trên waveform đã được tiền xử lý.
        """
        duration = len(clean_waveform) / sr if sr > 0 else 0.0

        try:
            # 1. Trích xuất đặc trưng Step 3 & Dự đoán xác suất
            feat_vec = self.feature_extractor.extract(clean_waveform)
            pred_dict = self.classifier.predict_single(feat_vec)
    
            classifier_ev = ClassifierEvidence(
                p_bonafide=pred_dict["prob_bonafide"],
                p_spoof=pred_dict["prob_spoof"],
                prediction=pred_dict["prediction"],
            )
    
            # 2. STFT cho Spectral Evidence
            padded_wave = clean_waveform
            if len(padded_wave) < self.n_fft:
                padded_wave = np.pad(padded_wave, (0, self.n_fft - len(padded_wave)))
    
            freqs, _, Zxx = signal.stft(
                padded_wave,
                fs=sr,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
                window="hann",
                boundary="zeros",
            )
            mag = np.abs(Zxx)
            pwr = mag**2
    
            spectral_ev = self._extract_spectral_evidence(mag, pwr, freqs)
            temporal_ev = self._extract_temporal_evidence(clean_waveform, sr)
            cepstral_ev = self._extract_cepstral_evidence(clean_waveform)
            prosodic_ev = self._extract_prosodic_evidence(clean_waveform, sr)
    
            # 3. Acoustic Challenges & Stability Analysis
            if run_challenges:
                runner = ChallengeRunner(seed=challenge_seed)
                challenge_results, stability_ev = runner.run(
                    waveform=clean_waveform,
                    sr=sr,
                    classifier=self.classifier,
                    feature_extractor=self.feature_extractor,
                    original_p_spoof=classifier_ev.p_spoof,
                    original_feature_vector=feat_vec,
                )
            else:
                challenge_results = []
                from .evidence import StabilityEvidence
                stability_ev = StabilityEvidence(
                    acoustic_consistency_score=1.0,
                    challenge_count=0,
                    mean_abs_delta_p_spoof=0.0,
                    mean_evidence_delta=0.0,
                )
    
            metadata_ev = MetadataEvidence(
                sample_rate=sr,
                duration_seconds=round(duration, 4),
                processing_status="success",
            )
    
            return AcousticChainOfEvidence(
                file_id=file_id,
                classifier=classifier_ev,
                spectral=spectral_ev,
                temporal=temporal_ev,
                cepstral=cepstral_ev,
                prosodic=prosodic_ev,
                challenges=challenge_results,
                stability=stability_ev,
                metadata=metadata_ev,
            )
        except Exception as e:
            # Handle failure cases by returning a failed evidence chain
            return AcousticChainOfEvidence(
                file_id=file_id,
                classifier=ClassifierEvidence(0.0, 0.0, "unknown"),
                spectral=SpectralEvidence(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),
                temporal=TemporalEvidence(0.0,0.0,0.0,0.0,0.0,0.0,0.0),
                cepstral=CepstralEvidence([],[],[],[],[],[]),
                prosodic=ProsodicEvidence("unavailable"),
                challenges=[],
                stability=StabilityEvidence(1.0,0,0.0,0.0),
                metadata=MetadataEvidence(
                    sample_rate=sr, 
                    duration_seconds=round(duration, 4), 
                    processing_status="failed", 
                    error_message=str(e)
                )
            )

    def analyze_file(
        self,
        audio_path: str | Path,
        run_challenges: bool = True,
        challenge_seed: int = 42,
    ) -> AcousticChainOfEvidence:
        """
        Đọc file từ đĩa, tiền xử lý qua Step 2, và phân tích ACoE đầy đủ.
        BẢO ĐẢM: Không bao giờ sửa đổi file gốc.
        """
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"File audio không tồn tại: {p}")

        # Tiền xử lý Step 2 (hoàn toàn trong bộ nhớ RAM)
        prep_res = self.preprocessor.process_file(p, output_dir=None)
        clean_waveform = prep_res["processed_waveform"]
        sr = prep_res["target_sr"]

        return self.analyze_waveform(
            clean_waveform=clean_waveform,
            sr=sr,
            file_id=p.stem,
            run_challenges=run_challenges,
            challenge_seed=challenge_seed,
        )
