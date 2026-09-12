#!/usr/bin/env python3
"""
src/analysis/acoustic_challenges.py

Thực thi các thử thách âm học có kiểm soát (Controlled Acoustic Challenges / Perturbations):
Mục đích:
  Đo lường độ ổn định của quyết định mô hình và các chứng cứ âm học dưới các biến đổi nhẹ, thực tế:
  1. Gain Perturbation (Biến đổi âm lượng nhẹ: +/- 10% hoặc +/- 1 dB)
  2. Low-Level Additive Noise (Thêm nhiễu trắng mức thấp, SNR = 35 dB)
  3. Small Resampling Perturbation (Biến đổi tần số lấy mẫu nhẹ: 16k -> 15.2k -> 16k mô phỏng jitter codec)

Quy tắc bắt buộc:
  - Không bao giờ ghi đè lên file âm thanh gốc.
  - Mang tính tất định (Deterministic) khi cố định random seed.
  - Tính toán Acoustic Consistency Score trong đoạn [0, 1].
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import signal

from .evidence import ChallengeResult, StabilityEvidence


class BaseChallenge:
    """Lớp cơ sở cho các thử thách âm học."""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed

    def apply(self, waveform: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError


class GainChallenge(BaseChallenge):
    """
    Thử thách 1: Biến đổi âm lượng (Gain Perturbation).
    Nhân biên độ với hệ số gain nhẹ (ví dụ 0.90 hoặc 1.10).
    """

    def __init__(self, gain_factor: float = 0.90, seed: int = 42):
        if gain_factor <= 0:
            raise ValueError("gain_factor must be > 0")
        super().__init__(name=f"gain_{gain_factor:.2f}", seed=seed)
        self.gain_factor = gain_factor

    def apply(self, waveform: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        perturbed = np.clip(waveform * self.gain_factor, -1.0, 1.0).astype(np.float32)
        gain_db = 20.0 * math.log10(self.gain_factor)
        return perturbed, {"gain_factor": self.gain_factor, "gain_db": round(gain_db, 2)}


class AdditiveNoiseChallenge(BaseChallenge):
    """
    Thử thách 2: Thêm nhiễu trắng mức thấp (Low-level Additive White Gaussian Noise).
    Mô phỏng nhiễu kênh truyền phòng thu / đường truyền với SNR cao (mặc định 35 dB).
    """

    def __init__(self, snr_db: float = 35.0, seed: int = 42):
        super().__init__(name=f"noise_snr_{snr_db:.0f}db", seed=seed)
        self.snr_db = snr_db

    def apply(self, waveform: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        rng = np.random.RandomState(self.seed)
        sig_power = np.mean(waveform**2)

        if sig_power < 1e-9:
            # Tín hiệu gần như im lặng
            noise = (rng.randn(*waveform.shape) * 1e-5).astype(np.float32)
        else:
            noise_power = sig_power / (10.0 ** (self.snr_db / 10.0))
            noise = (rng.randn(*waveform.shape) * np.sqrt(noise_power)).astype(np.float32)

        perturbed = np.clip(waveform + noise, -1.0, 1.0).astype(np.float32)
        return perturbed, {"snr_db": self.snr_db, "seed": self.seed}


class ResamplingChallenge(BaseChallenge):
    """
    Thử thách 3: Biến đổi tần số lấy mẫu nhẹ (Small Resampling Perturbation).
    Mô phỏng jitter codec hoặc đổi sampling rate nhẹ: 16000 -> intermediate_sr -> 16000.
    """

    def __init__(self, intermediate_sr: int = 15200, seed: int = 42):
        super().__init__(name=f"resampling_{intermediate_sr}hz", seed=seed)
        self.intermediate_sr = intermediate_sr

    def apply(self, waveform: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        orig_len = len(waveform)
        if orig_len == 0:
            return waveform, {"intermediate_sr": self.intermediate_sr}

        # 16000 -> intermediate_sr
        gcd1 = math.gcd(self.intermediate_sr, sr)
        up1 = self.intermediate_sr // gcd1
        down1 = sr // gcd1
        downsampled = signal.resample_poly(waveform, up1, down1).astype(np.float32)

        # intermediate_sr -> 16000
        gcd2 = math.gcd(sr, self.intermediate_sr)
        up2 = sr // gcd2
        down2 = self.intermediate_sr // gcd2
        restored = signal.resample_poly(downsampled, up2, down2).astype(np.float32)

        # Cắt hoặc đệm cho đúng độ dài gốc
        if len(restored) > orig_len:
            restored = restored[:orig_len]
        elif len(restored) < orig_len:
            restored = np.pad(restored, (0, orig_len - len(restored)))

        return restored, {"intermediate_sr": self.intermediate_sr, "target_sr": sr}


class ChallengeRunner:
    """
    Bộ điều phối chạy tập hợp các thử thách âm học và tính điểm ổn định (Stability Analysis).
    """

    def __init__(self, challenges: List[BaseChallenge] = None, seed: int = 42):
        if challenges is None:
            self.challenges = [
                GainChallenge(gain_factor=0.90, seed=seed),
                GainChallenge(gain_factor=1.10, seed=seed),
                AdditiveNoiseChallenge(snr_db=35.0, seed=seed),
                ResamplingChallenge(intermediate_sr=15200, seed=seed),
            ]
        else:
            self.challenges = challenges

    def run(
        self,
        waveform: np.ndarray,
        sr: int,
        classifier,
        feature_extractor,
        original_p_spoof: float,
        original_feature_vector: np.ndarray,
    ) -> Tuple[List[ChallengeResult], StabilityEvidence]:
        """
        Chạy toàn bộ các thử thách trên waveform, ghi nhận sự thay đổi của xác suất và đặc trưng.
        """
        results: List[ChallengeResult] = []
        delta_p_list: List[float] = []
        evidence_delta_list: List[float] = []
        per_challenge_stab: Dict[str, float] = {}

        orig_feat_norm = float(np.linalg.norm(original_feature_vector))
        if orig_feat_norm < 1e-5:
            orig_feat_norm = 1.0

        for ch in self.challenges:
            perturbed_wave, params = ch.apply(waveform, sr)

            # Trích xuất đặc trưng trên waveform bị thử thách
            ch_feat_vec = feature_extractor.extract(perturbed_wave)
            probs = classifier.predict_proba(np.atleast_2d(ch_feat_vec))[0]
            ch_p_spoof = float(probs[1])

            delta_p = ch_p_spoof - original_p_spoof
            feat_diff_norm = float(np.linalg.norm(ch_feat_vec - original_feature_vector))
            evidence_delta = feat_diff_norm / orig_feat_norm

            # Độ ổn định cục bộ của thử thách (1.0 - |delta_p|)
            ch_stability = max(0.0, 1.0 - abs(delta_p))
            per_challenge_stab[ch.name] = round(ch_stability, 4)

            results.append(
                ChallengeResult(
                    name=ch.name,
                    parameters=params,
                    p_spoof=round(ch_p_spoof, 4),
                    delta_p_spoof=round(delta_p, 4),
                    evidence_delta=round(evidence_delta, 4),
                )
            )
            delta_p_list.append(abs(delta_p))
            evidence_delta_list.append(evidence_delta)

        # Tính điểm nhất quán âm học tổng hợp (Acoustic Consistency Score)
        # Công thức tất định:
        #   score = max(0.0, 1.0 - mean(|delta_p|) - 0.2 * mean(evidence_delta))
        # Bounded trong [0, 1].
        mean_abs_delta_p = float(np.mean(delta_p_list)) if delta_p_list else 0.0
        mean_ev_delta = float(np.mean(evidence_delta_list)) if evidence_delta_list else 0.0

        raw_score = 1.0 - mean_abs_delta_p - 0.2 * mean_ev_delta
        consistency_score = round(max(0.0, min(1.0, float(raw_score))), 4)

        stability = StabilityEvidence(
            acoustic_consistency_score=consistency_score,
            challenge_count=len(self.challenges),
            mean_abs_delta_p_spoof=round(mean_abs_delta_p, 4),
            mean_evidence_delta=round(mean_ev_delta, 4),
            per_challenge_stability=per_challenge_stab,
        )

        return results, stability
