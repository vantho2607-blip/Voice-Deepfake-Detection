#!/usr/bin/env python3
"""
scripts/analyze_acoustic.py

Giao diện dòng lệnh (CLI) phân tích chuỗi chứng cứ âm học (Acoustic Chain of Evidence - ACoE)
cho một file âm thanh đơn lẻ.

Sử dụng:
    py -3.12 scripts/analyze_acoustic.py --audio <path_to_audio.flac>
    py -3.12 scripts/analyze_acoustic.py --audio <path_to_audio.flac> --output output.json
    py -3.12 scripts/analyze_acoustic.py --audio <path_to_audio.flac> --model data/processed/models/baseline_logistic_regression.joblib
"""

import argparse
from pathlib import Path
import sys

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.analysis.acoustic_analyzer import AcousticAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phân tích chuỗi chứng cứ âm học (ACoE) cho file âm thanh đơn lẻ."
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Đường dẫn đến file âm thanh cần phân tích (.flac, .wav).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/processed/models/baseline_logistic_regression.joblib",
        help="Đường dẫn checkpoint mô hình phân loại Step 3.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file .json để lưu kết quả cấu trúc ACoE (tùy chọn).",
    )
    parser.add_argument(
        "--no-challenges",
        action="store_true",
        help="Bỏ qua bước thử thách âm học có kiểm soát.",
    )
    return parser.parse_args()


def print_summary(evidence):
    c = evidence.classifier
    s = evidence.spectral
    t = evidence.temporal
    cep = evidence.cepstral
    p = evidence.prosodic
    st = evidence.stability
    m = evidence.metadata

    pred_tag = "🔴 SPOOF" if c.prediction.lower() == "spoof" else "🟢 BONAFIDE"

    print("\n" + "=" * 64)
    print("      ACOUSTIC CHAIN OF EVIDENCE (ACoE) — REPORT")
    print("=" * 64)
    print(f"File ID        : {evidence.file_id}")
    print(f"Sample Rate    : {m.sample_rate} Hz")
    print(f"Duration       : {m.duration_seconds:.2f} s")
    print(f"Status         : {m.processing_status}")

    print("\n--- [1] CLASSIFIER EVIDENCE (Step 3 Baseline) ---")
    print(f"Verdict        : {pred_tag}")
    print(f"P(bonafide)    : {c.p_bonafide:.4f}")
    print(f"P(spoof)       : {c.p_spoof:.4f}")

    print("\n--- [2] SPECTRAL EVIDENCE ---")
    print(f"Centroid       : {s.centroid_mean:.1f} ± {s.centroid_std:.1f} Hz")
    print(f"Rolloff (85%)  : {s.rolloff_mean:.1f} ± {s.rolloff_std:.1f} Hz")
    print(f"Bandwidth      : {s.bandwidth_mean:.1f} ± {s.bandwidth_std:.1f} Hz")
    print(f"Flatness       : {s.flatness_mean:.4f} ± {s.flatness_std:.4f}")
    print(f"Spectral Flux  : {s.flux_mean:.4f} ± {s.flux_std:.4f}")

    print("\n--- [3] TEMPORAL EVIDENCE ---")
    print(f"RMS Energy     : {t.rms_mean:.4f} ± {t.rms_std:.4f} (Var: {t.rms_variation:.4f})")
    print(f"Zero Crossing  : {t.zcr_mean:.4f} ± {t.zcr_std:.4f}")
    print(f"Silence Ratio  : {t.silence_ratio * 100:.1f}%")
    print(f"Voiced Est.    : {t.voiced_ratio_est * 100:.1f}%")

    print("\n--- [4] CEPSTRAL EVIDENCE (MFCC Summary) ---")
    mfcc_m_str = ", ".join(f"{v:.2f}" for v in cep.mfcc_mean[:6])
    print(f"MFCC Mean [0-5]: [{mfcc_m_str}, ...]")

    print("\n--- [5] PROSODIC EVIDENCE (F0 Tracking) ---")
    if p.status == "available":
        print(f"Status         : AVAILABLE")
        print(f"F0 Mean        : {p.f0_mean:.1f} ± {p.f0_std:.1f} Hz")
        print(f"F0 Range       : {p.f0_range:.1f} Hz")
        print(f"Voiced Ratio   : {p.voiced_ratio * 100:.1f}%")
        print(f"Pitch Variation: {p.pitch_variation:.4f}")
    else:
        print("Status         : UNAVAILABLE (Không đủ thông tin pitch tin cậy hoặc âm thanh im lặng)")

    print("\n--- [6] CONTROLLED ACOUSTIC CHALLENGES & STABILITY ---")
    if evidence.challenges:
        print(f"{'Challenge Name':<22} | {'P(spoof)':<10} | {'Delta P':<10} | {'Evidence Delta':<14}")
        print("-" * 64)
        for ch in evidence.challenges:
            d_p_str = f"{ch.delta_p_spoof:+.4f}"
            print(f"{ch.name:<22} | {ch.p_spoof:<10.4f} | {d_p_str:<10} | {ch.evidence_delta:<14.4f}")
        print("-" * 64)
        print(f"Acoustic Consistency Score : {st.acoustic_consistency_score:.4f} / 1.0000")
        print(f"Mean |Delta P(spoof)|       : {st.mean_abs_delta_p_spoof:.4f}")
        print(f"Mean Evidence Delta         : {st.mean_evidence_delta:.4f}")
    else:
        print("Bỏ qua thử thách âm học (--no-challenges).")

    print("=" * 64 + "\n")


def main():
    args = parse_args()
    audio_path = Path(args.audio)

    if not audio_path.exists():
        print(f"Lỗi: Không tìm thấy file âm thanh tại: {audio_path}", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Lỗi: Không tìm thấy checkpoint mô hình tại: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Đang khởi tạo AcousticAnalyzer từ model: {model_path} ...")
    analyzer = AcousticAnalyzer(model_path=model_path)

    print(f"Đang phân tích file: {audio_path.name} ...")
    evidence = analyzer.analyze_file(
        audio_path=audio_path,
        run_challenges=not args.no_challenges,
    )

    print_summary(evidence)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(evidence.to_json(indent=2))
        print(f"Đã lưu bản ghi ACoE vào: {out_path.resolve()}")


if __name__ == "__main__":
    main()
