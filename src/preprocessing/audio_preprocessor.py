#!/usr/bin/env python3
"""
src/preprocessing/audio_preprocessor.py

Module tiền xử lý và lọc nhiễu âm thanh cho dự án DAP391m:
Pipeline chuẩn gồm 5 bước theo đúng thứ tự:
  1. Load Audio & Validate
  2. Ensure 16 kHz (Resampling nếu cần)
  3. Silence Trimming (Chỉ cắt khoảng lặng ở đầu và cuối)
  4. Amplitude Normalization (Peak normalization an toàn, chống NaN/Inf)
  5. Spectral Subtraction (Lọc nhiễu phổ bằng thuật toán STFT - Subtraction - ISTFT)
  6. Save Processed Audio (Lưu file ngoài thư mục dữ liệu gốc)

Sử dụng thư viện: soundfile, numpy, scipy.
"""

from pathlib import Path
import warnings
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy import signal
import soundfile as sf


class AudioPreprocessor:
    """
    Bộ tiền xử lý âm thanh chuẩn hóa cho phát hiện Voice Cloning.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        trim_top_db: float = 25.0,
        trim_frame_length: int = 512,
        trim_hop_length: int = 128,
        norm_target_peak: float = 0.95,
        spec_n_fft: int = 512,
        spec_hop_length: int = 128,
        spec_alpha: float = 1.5,
        spec_beta: float = 0.02,
    ):
        self.target_sr = target_sr
        self.trim_top_db = trim_top_db
        self.trim_frame_length = trim_frame_length
        self.trim_hop_length = trim_hop_length
        self.norm_target_peak = norm_target_peak
        self.spec_n_fft = spec_n_fft
        self.spec_hop_length = spec_hop_length
        self.spec_alpha = spec_alpha
        self.spec_beta = spec_beta

    # -------------------------------------------------------------
    # BƯỚC 1: LOAD AUDIO
    # -------------------------------------------------------------
    @staticmethod
    def load_audio(file_path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Đọc file âm thanh từ đĩa (FLAC, WAV).
        Trả về (waveform_1d, sample_rate).
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File audio không tồn tại: {p}")

        data, sr = sf.read(str(p), dtype="float32")

        # Chuyển stereo -> mono nếu có
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        return data, sr

    # -------------------------------------------------------------
    # BƯỚC 2: ENSURE 16 KHZ (RESAMPLING NẾU CẦN)
    # -------------------------------------------------------------
    def ensure_sample_rate(
        self, waveform: np.ndarray, orig_sr: int
    ) -> Tuple[np.ndarray, int, bool]:
        """
        Kiểm tra và resample về target_sr nếu cần.
        Nếu orig_sr == target_sr: giữ nguyên, resampled=False.
        Nếu orig_sr != target_sr: resample đa thức bằng scipy.signal.resample_poly.
        """
        if orig_sr == self.target_sr:
            return waveform, self.target_sr, False

        # Tính tỷ lệ đơn giản hóa ước số chung lớn nhất (GCD)
        import math
        gcd = math.gcd(self.target_sr, orig_sr)
        up = self.target_sr // gcd
        down = orig_sr // gcd

        resampled_waveform = signal.resample_poly(waveform, up, down).astype(np.float32)
        return resampled_waveform, self.target_sr, True

    # -------------------------------------------------------------
    # BƯỚC 3: SILENCE TRIMMING (ĐẦU VÀ CUỐI)
    # -------------------------------------------------------------
    def trim_silence(
        self, waveform: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cắt tỉa khoảng lặng (leading & trailing silence).
        KHÔNG cắt khoảng lặng giữa các từ/câu.
        Sử dụng phân tích năng lượng ngắn hạn (Short-Time Frame RMS Energy).
        """
        duration_before = len(waveform) / sr
        if len(waveform) == 0:
            return waveform, {
                "top_db": self.trim_top_db,
                "frame_length": self.trim_frame_length,
                "hop_length": self.trim_hop_length,
                "duration_before": 0.0,
                "duration_after": 0.0,
                "trimmed": False,
            }

        num_samples = len(waveform)
        frame_len = self.trim_frame_length
        hop_len = self.trim_hop_length

        if num_samples < frame_len:
            # Audio quá ngắn so với frame, giữ nguyên
            return waveform, {
                "top_db": self.trim_top_db,
                "frame_length": frame_len,
                "hop_length": hop_len,
                "duration_before": duration_before,
                "duration_after": duration_before,
                "trimmed": False,
            }

        # Chia frame và tính RMS
        num_frames = 1 + (num_samples - frame_len) // hop_len
        shape = (num_frames, frame_len)
        strides = (waveform.strides[0] * hop_len, waveform.strides[0])
        frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides)

        frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        frame_db = 20.0 * np.log10(frame_rms + 1e-12)

        peak_db = np.max(frame_db)
        threshold_db = peak_db - self.trim_top_db

        non_silent = np.where(frame_db >= threshold_db)[0]
        if len(non_silent) == 0:
            # Toàn bộ tín hiệu coi như im lặng
            return waveform, {
                "top_db": self.trim_top_db,
                "frame_length": frame_len,
                "hop_length": hop_len,
                "duration_before": duration_before,
                "duration_after": duration_before,
                "trimmed": False,
            }

        start_frame = non_silent[0]
        end_frame = non_silent[-1]

        start_sample = max(0, start_frame * hop_len)
        end_sample = min(num_samples, end_frame * hop_len + frame_len)

        trimmed_waveform = waveform[start_sample:end_sample]
        duration_after = len(trimmed_waveform) / sr

        return trimmed_waveform, {
            "top_db": self.trim_top_db,
            "frame_length": frame_len,
            "hop_length": hop_len,
            "duration_before": round(duration_before, 4),
            "duration_after": round(duration_after, 4),
            "samples_before": num_samples,
            "samples_after": len(trimmed_waveform),
            "trimmed": (len(trimmed_waveform) < num_samples),
        }

    # -------------------------------------------------------------
    # BƯỚC 4: AMPLITUDE NORMALIZATION
    # -------------------------------------------------------------
    def normalize_amplitude(
        self, waveform: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Chuẩn hóa biên độ đỉnh (Peak Normalization) về target_peak (mặc định 0.95).
        Bảo đảm an toàn: không chia cho 0, không tạo NaN, không tạo Inf.
        """
        if len(waveform) == 0:
            return waveform, {"peak_before": 0.0, "peak_after": 0.0, "normalized": False}

        peak_before = float(np.max(np.abs(waveform)))

        if peak_before == 0.0 or not np.isfinite(peak_before):
            # Tín hiệu hoàn toàn rỗng hoặc zero
            return waveform, {
                "peak_before": peak_before,
                "peak_after": 0.0,
                "normalized": False,
            }

        scale = self.norm_target_peak / peak_before
        normalized_waveform = waveform * scale

        peak_after = float(np.max(np.abs(normalized_waveform)))
        return normalized_waveform.astype(np.float32), {
            "peak_before": round(peak_before, 6),
            "peak_after": round(peak_after, 6),
            "target_peak": self.norm_target_peak,
            "normalized": True,
        }

    # -------------------------------------------------------------
    # BƯỚC 5: SPECTRAL SUBTRACTION (LỌC NHIỄU PHỔ)
    # -------------------------------------------------------------
    def spectral_subtraction(
        self, waveform: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Thuật toán lọc nhiễu Spectral Subtraction (Berouti / Boll):
          waveform -> STFT -> Noise spectrum estimation -> Magnitude Subtraction
          -> Phase Preservation -> ISTFT -> waveform.

        Phương pháp ước lượng phổ nhiễu:
          Do là single-channel speech không có kênh noise riêng, phổ nhiễu dừng N(f)
          được ước lượng qua bách phân vị thứ 10 (10th percentile) theo trục thời gian
          của từng dải tần số (heuristic tối ưu cho speech enhancement).
        """
        if len(waveform) < self.spec_n_fft:
            # Audio quá ngắn để STFT
            return waveform, {
                "n_fft": self.spec_n_fft,
                "hop_length": self.spec_hop_length,
                "applied": False,
                "reason": "waveform shorter than n_fft",
            }

        peak_before = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0
        rms_before = float(np.sqrt(np.mean(waveform**2))) if len(waveform) > 0 else 0.0

        n_fft = self.spec_n_fft
        hop_len = self.spec_hop_length
        noverlap = n_fft - hop_len

        # 1. STFT
        f, t, Zxx = signal.stft(
            waveform,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=noverlap,
            boundary="zeros",
            padded=True,
        )

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        power = magnitude**2

        # 2. Ước lượng phổ nhiễu (Heuristic: 10th percentile across time frames)
        noise_magnitude = np.percentile(magnitude, 10, axis=1, keepdims=True)
        noise_power = noise_magnitude**2

        # 3. Trừ phổ công suất (Power Spectral Subtraction với over-subtraction & flooring)
        # P_sub = max(P - alpha * P_noise, beta^2 * P)
        alpha = self.spec_alpha
        beta = self.spec_beta

        subtracted_power = power - alpha * noise_power
        floor_power = (beta**2) * power
        cleaned_power = np.maximum(subtracted_power, floor_power)
        cleaned_magnitude = np.sqrt(cleaned_power)

        # 4. Tái lập phổ phức (Bảo tồn pha gốc)
        Zxx_cleaned = cleaned_magnitude * np.exp(1j * phase)

        # 5. ISTFT
        _, x_denoised = signal.istft(
            Zxx_cleaned,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=noverlap,
            boundary="zeros",
        )

        # Cắt cho đúng độ dài ban đầu
        orig_len = len(waveform)
        if len(x_denoised) > orig_len:
            x_denoised = x_denoised[:orig_len]
        elif len(x_denoised) < orig_len:
            x_denoised = np.pad(x_denoised, (0, orig_len - len(x_denoised)))

        # Kiểm tra NaN / Inf
        if not np.all(np.isfinite(x_denoised)):
            warnings.warn("Phát hiện giá trị không hợp lệ trong spectral subtraction, khôi phục waveform gốc.")
            x_denoised = waveform.copy()

        x_denoised = x_denoised.astype(np.float32)
        peak_after = float(np.max(np.abs(x_denoised)))
        rms_after = float(np.sqrt(np.mean(x_denoised**2)))

        return x_denoised, {
            "n_fft": n_fft,
            "hop_length": hop_len,
            "alpha": alpha,
            "beta": beta,
            "noise_estimation_method": "10th-percentile time-frequency energy tracker (heuristic)",
            "peak_before": round(peak_before, 6),
            "peak_after": round(peak_after, 6),
            "rms_before": round(rms_before, 6),
            "rms_after": round(rms_after, 6),
            "applied": True,
        }

    # -------------------------------------------------------------
    # BƯỚC 6: SAVE AUDIO
    # -------------------------------------------------------------
    @staticmethod
    def save_audio(
        waveform: np.ndarray,
        sr: int,
        output_path: str | Path,
        raw_data_dir: Optional[str | Path] = None,
    ) -> Path:
        """
        Lưu file audio ra đĩa an toàn.
        BẢO ĐẢM: không cho phép ghi đè vào thư mục dữ liệu gốc.
        """
        out_p = Path(output_path).resolve()

        if raw_data_dir is None:
            raw_data_dir = Path(__file__).resolve().parent.parent.parent / "data"

        raw_p = Path(raw_data_dir).resolve()
        try:
            out_p.relative_to(raw_p)
            raise PermissionError(
                f"NGHIÊM CẤM GHI ĐÈ: Đường dẫn output {out_p} nằm trong thư mục dữ liệu gốc {raw_p}!"
            )
        except ValueError:
            pass

        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), waveform, sr, subtype="PCM_16")
        return out_p

    # -------------------------------------------------------------
    # FULL PIPELINE EXECUTION (THEO ĐÚNG THỨ TỰ BẮT BUỘC)
    # -------------------------------------------------------------
    def process_file(
        self,
        audio_path: str | Path,
        output_dir: Optional[str | Path] = None,
        save_format: str = "wav",
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline theo thứ tự:
          Load -> Ensure 16kHz -> Silence Trim -> Normalize -> Spectral Subtraction -> Save
        """
        audio_path = Path(audio_path)

        # 1. Load
        raw_waveform, orig_sr = self.load_audio(audio_path)
        orig_peak = float(np.max(np.abs(raw_waveform))) if len(raw_waveform) > 0 else 0.0
        orig_rms = float(np.sqrt(np.mean(raw_waveform**2))) if len(raw_waveform) > 0 else 0.0
        orig_duration = len(raw_waveform) / orig_sr if orig_sr > 0 else 0.0

        # 2. Ensure 16 kHz
        resampled_waveform, current_sr, was_resampled = self.ensure_sample_rate(
            raw_waveform, orig_sr
        )

        # 3. Silence Trim
        trimmed_waveform, trim_info = self.trim_silence(resampled_waveform, current_sr)

        # 4. Amplitude Normalization
        norm_waveform, norm_info = self.normalize_amplitude(trimmed_waveform)

        # 5. Spectral Subtraction
        denoised_waveform, spec_info = self.spectral_subtraction(norm_waveform, current_sr)

        # 6. Save (nếu có output_dir)
        saved_path = None
        if output_dir is not None:
            out_dir_p = Path(output_dir)
            out_filename = f"{audio_path.stem}_processed.{save_format}"
            saved_path = self.save_audio(
                denoised_waveform, current_sr, out_dir_p / out_filename
            )

        final_peak = float(np.max(np.abs(denoised_waveform))) if len(denoised_waveform) > 0 else 0.0
        final_rms = float(np.sqrt(np.mean(denoised_waveform**2))) if len(denoised_waveform) > 0 else 0.0
        final_duration = len(denoised_waveform) / current_sr if current_sr > 0 else 0.0

        return {
            "file_id": audio_path.stem,
            "input_path": str(audio_path),
            "output_path": str(saved_path) if saved_path else None,
            "original_sr": orig_sr,
            "target_sr": current_sr,
            "resampled": was_resampled,
            "original_duration": round(orig_duration, 4),
            "final_duration": round(final_duration, 4),
            "original_peak": round(orig_peak, 4),
            "final_peak": round(final_peak, 4),
            "original_rms": round(orig_rms, 4),
            "final_rms": round(final_rms, 4),
            "trim_info": trim_info,
            "norm_info": norm_info,
            "spec_info": spec_info,
            "has_nan": bool(np.isnan(denoised_waveform).any()),
            "has_inf": bool(np.isinf(denoised_waveform).any()),
            "processed_waveform": denoised_waveform,
        }
