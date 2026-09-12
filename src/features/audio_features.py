#!/usr/bin/env python3
"""
src/features/audio_features.py

Module trích xuất đặc trưng âm học (Acoustic Feature Extraction) cho BƯỚC 3:
Trích xuất vector đặc trưng cố định chiều từ tín hiệu âm thanh đã qua tiền xử lý (BƯỚC 2):
  1. MFCC (20 hệ số)
  2. Delta MFCC (20 hệ số)
  3. Delta-Delta MFCC (20 hệ số)
  4. Spectral Centroid (Trọng tâm phổ)
  5. Spectral Rolloff (Tần số cuộn phổ 85%)
  6. Zero-Crossing Rate (Tỷ lệ đổi dấu)
  7. RMS Energy (Năng lượng hiệu dụng)

Cơ chế Pooling:
  Áp dụng thống kê Mean và Standard Deviation trên toàn bộ các frame thời gian:
  - 20 MFCC * 2 (mean, std) = 40 chiều
  - 20 Delta * 2 (mean, std) = 40 chiều
  - 20 Delta-Delta * 2 (mean, std) = 40 chiều
  - 4 Spectral/Temporal features * 2 (mean, std) = 8 chiều
  Tổng số chiều cố định: 128 chiều.
"""

from typing import List, Tuple
import numpy as np
from scipy import fft, signal


def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """Chuyển đổi tần số Hertz sang Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Chuyển đổi Mel scale sang Hertz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class AcousticFeatureExtractor:
    """
    Bộ trích xuất đặc trưng âm học chuẩn 128 chiều cho phát hiện Voice Cloning.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 20,
        fmin: float = 20.0,
        fmax: float = 8000.0,
        delta_order: int = 2,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.delta_order = delta_order

        # Khởi tạo ma trận Mel filterbank
        self.fbank = self._build_mel_filterbank()
        self._feature_names = self._generate_feature_names()

    def _build_mel_filterbank(self) -> np.ndarray:
        """Tạo ma trận bộ lọc tam giác Mel (Mel Triangular Filterbank)."""
        # Đã nâng cấp từ Mel (MFCC) lên Linear (LFCC) để bắt lỗi cao tần
        hz_points = np.linspace(self.fmin, self.fmax, self.n_mels + 2)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sr).astype(int)

        num_bins = self.n_fft // 2 + 1
        fbank = np.zeros((self.n_mels, num_bins), dtype=np.float32)

        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return fbank

    def _compute_deltas(self, c: np.ndarray) -> np.ndarray:
        """
        Tính đạo hàm bậc 1/2 theo thời gian (Delta / Delta-Delta).
        Công thức chuẩn ASVspoof:
          Delta[t] = sum(n * (c[t+n] - c[t-n])) / (2 * sum(n^2))
        """
        order = self.delta_order
        pad_c = np.pad(c, ((order, order), (0, 0)), mode="edge")
        delta = np.zeros_like(c)
        denom = 2 * sum(n**2 for n in range(1, order + 1))

        for n in range(1, order + 1):
            delta += n * (pad_c[order + n : len(pad_c) - order + n] - pad_c[order - n : len(pad_c) - order - n])

        return delta / denom

    def _generate_feature_names(self) -> List[str]:
        """Danh sách 128 tên đặc trưng để minh bạch hóa mô hình."""
        names = []
        for i in range(self.n_mels):
            names.append(f"lfcc_{i}_mean")
        for i in range(self.n_mels):
            names.append(f"lfcc_{i}_std")
        for i in range(self.n_mels):
            names.append(f"delta1_{i}_mean")
        for i in range(self.n_mels):
            names.append(f"delta1_{i}_std")
        for i in range(self.n_mels):
            names.append(f"delta2_{i}_mean")
        for i in range(self.n_mels):
            names.append(f"delta2_{i}_std")
        names.extend(["spec_centroid_mean", "spec_centroid_std"])
        names.extend(["spec_rolloff_mean", "spec_rolloff_std"])
        names.extend(["zcr_mean", "zcr_std"])
        names.extend(["rms_energy_mean", "rms_energy_std"])
        return names

    @property
    def feature_dim(self) -> int:
        return len(self._feature_names)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """
        Trích xuất vector 128 chiều từ mảng waveform 1D.
        Đảm bảo an toàn số học (không có NaN, không có Inf).
        """
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Padding nếu audio ngắn hơn kích thước cửa sổ FFT
        if len(waveform) < self.n_fft:
            waveform = np.pad(waveform, (0, self.n_fft - len(waveform)))

        # 1. Tính biến đổi Fourier ngắn hạn (STFT)
        f, t, Zxx = signal.stft(
            waveform,
            fs=self.sr,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            window="hann",
            boundary="zeros",
            padded=True,
        )

        magnitude = np.abs(Zxx)
        power = magnitude**2

        # 2. Mel Filterbank Energy & MFCC
        mel_energies = np.dot(self.fbank, power)
        log_mel_energies = np.log(mel_energies + 1e-10)
        # DCT-II qua trục mel bins
        mfcc = fft.dct(log_mel_energies, type=2, axis=0, norm="ortho").T  # Shape: (T, n_mels)

        # 3. Delta và Delta-Delta MFCC
        delta1 = self._compute_deltas(mfcc)
        delta2 = self._compute_deltas(delta1)

        # 4. Spectral Centroid
        freq_weights = f[:, None]
        spec_centroid = np.sum(freq_weights * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-10)

        # 5. Spectral Rolloff (85% ngưỡng năng lượng tích lũy)
        cum_power = np.cumsum(power, axis=0)
        rolloff_indices = np.apply_along_axis(
            lambda col: np.searchsorted(col, 0.85 * col[-1]), 0, cum_power
        )
        rolloff_indices = np.clip(rolloff_indices, 0, len(f) - 1)
        spec_rolloff = f[rolloff_indices]

        # 6. Zero-Crossing Rate & RMS theo frame thời gian
        num_frames = magnitude.shape[1]
        zcr = np.zeros(num_frames, dtype=np.float32)
        rms = np.zeros(num_frames, dtype=np.float32)

        # Pad waveform to match STFT boundary="zeros" behavior
        pad_width = self.n_fft // 2
        padded_waveform = np.pad(waveform, (pad_width, pad_width), mode='constant')

        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            chunk = padded_waveform[start:end]
            if len(chunk) > 1:
                zcr[i] = ((chunk[:-1] * chunk[1:]) < 0).mean()
                rms[i] = np.sqrt(np.mean(chunk**2))

        # 7. Pooling: Mean & Std qua toàn bộ các frame
        feature_vector = np.hstack([
            np.mean(mfcc, axis=0),
            np.std(mfcc, axis=0),
            np.mean(delta1, axis=0),
            np.std(delta1, axis=0),
            np.mean(delta2, axis=0),
            np.std(delta2, axis=0),
            [np.mean(spec_centroid), np.std(spec_centroid)],
            [np.mean(spec_rolloff), np.std(spec_rolloff)],
            [np.mean(zcr), np.std(zcr)],
            [np.mean(rms), np.std(rms)],
        ]).astype(np.float32)

        # Kiểm tra an toàn số học
        if not np.all(np.isfinite(feature_vector)):
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_vector
