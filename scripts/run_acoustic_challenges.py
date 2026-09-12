#!/usr/bin/env python3
"""
scripts/run_acoustic_challenges.py

Thực thi đánh giá chuỗi chứng cứ âm học (ACoE) và các thử thách âm học có kiểm soát trên tập mẫu lớn:
- So sánh hiệu năng và độ ổn định giữa:
    + Bonafide (Âm thanh thật)
    + Known Attacks (A01 - A06: Các thuật toán spoofing đã thấy trong tập train/dev)
    + Unseen Attacks (A07 - A19: Các thuật toán spoofing mới hoàn toàn trong tập eval)
- Đánh giá tác động của từng loại biến đổi (Gain +/-10%, Additive Noise 35dB, Resampling 15.2kHz).
- Tính tỷ lệ đảo ngược quyết định (Flip Rate) và Điểm nhất quán âm học (Acoustic Consistency Score).
- Tự động vẽ và lưu 5 biểu đồ trực quan vào docs/figures/:
    1. docs/figures/p_spoof_distribution.png
    2. docs/figures/consistency_score_distribution.png
    3. docs/figures/known_vs_unseen_comparison.png
    4. docs/figures/per_attack_recall.png
    5. docs/figures/challenge_delta_p_spoof.png
- Lưu kết quả số liệu chi tiết vào docs/step4_challenge_metrics.json.
"""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Chế độ headless không cần màn hình GUI
import matplotlib.pyplot as plt
import numpy as np

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.analysis.acoustic_analyzer import AcousticAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá hàng loạt Acoustic Challenges & ACoE trên Known vs Unseen attacks."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/processed/metadata.db",
        help="Đường dẫn đến cơ sở dữ liệu SQLite metadata.db.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/processed/models/baseline_logistic_regression.joblib",
        help="Đường dẫn đến checkpoint mô hình phân loại Step 3.",
    )
    parser.add_argument(
        "--samples-per-attack",
        type=int,
        default=20,
        help="Số lượng mẫu đánh giá cho mỗi loại tấn công (mặc định 20 mẫu x 19 attacks = 380).",
    )
    parser.add_argument(
        "--bonafide-samples",
        type=int,
        default=50,
        help="Số lượng mẫu bonafide cho mỗi split (dev và eval, mặc định 50 mẫu mỗi split).",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="docs/figures",
        help="Thư mục lưu các biểu đồ báo cáo.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="docs/step4_challenge_metrics.json",
        help="Đường dẫn lưu kết quả số liệu dạng JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed cho lấy mẫu và thử thách.",
    )
    return parser.parse_args()


import random

def load_dataset_samples(db_path: Path, samples_per_attack: int, bonafide_samples: int, seed: int) -> List[Dict[str, Any]]:
    """Truy vấn lấy tập mẫu cân bằng từ SQLite metadata.db."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    rng = random.Random(seed)

    selected_records = []

    # 1. Dev Bonafide
    q_dev_bonafide = """
        SELECT file_id, speaker_id, split, label, attack_id, audio_path
        FROM samples
        WHERE split = 'dev' AND label = 'bonafide'
    """
    rows = cursor.execute(q_dev_bonafide).fetchall()
    rng.shuffle(rows)
    for row in rows[:bonafide_samples]:
        selected_records.append({
            "file_id": row[0],
            "speaker_id": row[1],
            "split": row[2],
            "label": row[3],
            "attack_id": "bonafide",
            "group": "Bonafide (Dev)",
            "audio_path": row[5],
        })

    # 2. Eval Bonafide
    q_eval_bonafide = """
        SELECT file_id, speaker_id, split, label, attack_id, audio_path
        FROM samples
        WHERE split = 'eval' AND label = 'bonafide'
    """
    rows = cursor.execute(q_eval_bonafide).fetchall()
    rng.shuffle(rows)
    for row in rows[:bonafide_samples]:
        selected_records.append({
            "file_id": row[0],
            "speaker_id": row[1],
            "split": row[2],
            "label": row[3],
            "attack_id": "bonafide",
            "group": "Bonafide (Eval)",
            "audio_path": row[5],
        })

    # 3. Known Attacks (Dev split: A01 -> A06)
    known_attacks = [f"A{i:02d}" for i in range(1, 7)]
    for att in known_attacks:
        q_known = """
            SELECT file_id, speaker_id, split, label, attack_id, audio_path
            FROM samples
            WHERE split = 'dev' AND label = 'spoof' AND attack_id = ?
        """
        rows = cursor.execute(q_known, (att,)).fetchall()
        rng.shuffle(rows)
        for row in rows[:samples_per_attack]:
            selected_records.append({
                "file_id": row[0],
                "speaker_id": row[1],
                "split": row[2],
                "label": row[3],
                "attack_id": att,
                "group": "Known Attacks (A01-A06)",
                "audio_path": row[5],
            })

    # 4. Unseen Attacks (Eval split: A07 -> A19)
    unseen_attacks = [f"A{i:02d}" for i in range(7, 20)]
    for att in unseen_attacks:
        q_unseen = """
            SELECT file_id, speaker_id, split, label, attack_id, audio_path
            FROM samples
            WHERE split = 'eval' AND label = 'spoof' AND attack_id = ?
        """
        rows = cursor.execute(q_unseen, (att,)).fetchall()
        rng.shuffle(rows)
        for row in rows[:samples_per_attack]:
            selected_records.append({
                "file_id": row[0],
                "speaker_id": row[1],
                "split": row[2],
                "label": row[3],
                "attack_id": att,
                "group": "Unseen Attacks (A07-A19)",
                "audio_path": row[5],
            })

    conn.close()
    return selected_records


def run_batch_evaluation(analyzer: AcousticAnalyzer, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chạy phân tích ACoE và Acoustic Challenges trên toàn bộ danh sách mẫu."""
    results = []
    total = len(samples)
    print(f"Bắt đầu phân tích {total} mẫu audio qua ACoE Pipeline...")

    for i, s in enumerate(samples, 1):
        if i % 50 == 0 or i == total:
            print(f"  -> Tiến độ: {i}/{total} ({i*100//total}%)")

        audio_path = Path(s["audio_path"])
        if not audio_path.exists():
            continue

        evidence = analyzer.analyze_file(audio_path, run_challenges=True)

        orig_p_spoof = evidence.classifier.p_spoof
        orig_pred = "spoof" if orig_p_spoof >= 0.80 else "bonafide"

        # Kiểm tra sự ổn định của quyết định mô hình (flip rate)
        ch_deltas = {}
        flips = 0
        for ch in evidence.challenges:
            ch_deltas[ch.name] = {
                "delta_p": ch.delta_p_spoof,
                "evidence_delta": ch.evidence_delta,
                "p_spoof": ch.p_spoof,
            }
            ch_pred = "spoof" if ch.p_spoof >= 0.80 else "bonafide"
            if ch_pred != orig_pred:
                flips += 1

        results.append({
            "file_id": s["file_id"],
            "split": s["split"],
            "label": s["label"],
            "attack_id": s["attack_id"],
            "group": s["group"],
            "p_bonafide": evidence.classifier.p_bonafide,
            "p_spoof": orig_p_spoof,
            "prediction": orig_pred,
            "is_correct": (orig_pred == s["label"]),
            "consistency_score": evidence.stability.acoustic_consistency_score,
            "mean_abs_delta_p": evidence.stability.mean_abs_delta_p_spoof,
            "mean_evidence_delta": evidence.stability.mean_evidence_delta,
            "challenge_deltas": ch_deltas,
            "flips": flips,
            "flip_rate": flips / max(1, len(evidence.challenges)),
            "duration": evidence.metadata.duration_seconds,
            "f0_status": evidence.prosodic.status,
            "f0_mean": evidence.prosodic.f0_mean,
            "centroid_mean": evidence.spectral.centroid_mean,
            "rms_mean": evidence.temporal.rms_mean,
        })

    return results


def compute_group_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tính toán thống kê theo nhóm và từng attack."""
    groups = {}
    for r in results:
        g = r["group"]
        groups.setdefault(g, []).append(r)

    group_summary = {}
    for g_name, items in groups.items():
        p_spoof_vals = [it["p_spoof"] for it in items]
        cs_vals = [it["consistency_score"] for it in items]
        delta_p_vals = [it["mean_abs_delta_p"] for it in items]
        ev_d_vals = [it["mean_evidence_delta"] for it in items]
        flips = [it["flips"] for it in items]
        correct = [1 if it["is_correct"] else 0 for it in items]

        # Recall (cho spoof) hoặc Accuracy (cho bonafide)
        acc_recall = float(np.mean(correct)) if correct else 0.0

        group_summary[g_name] = {
            "count": len(items),
            "accuracy_or_recall": round(acc_recall * 100, 2),
            "mean_p_spoof": round(float(np.mean(p_spoof_vals)), 4),
            "std_p_spoof": round(float(np.std(p_spoof_vals)), 4),
            "mean_consistency_score": round(float(np.mean(cs_vals)), 4),
            "std_consistency_score": round(float(np.std(cs_vals)), 4),
            "mean_abs_delta_p": round(float(np.mean(delta_p_vals)), 4),
            "mean_evidence_delta": round(float(np.mean(ev_d_vals)), 4),
            "mean_flips_per_sample": round(float(np.mean(flips)), 3),
        }

    # Thống kê chi tiết per attack (A01 -> A19)
    attack_summary = {}
    all_attacks = [f"A{i:02d}" for i in range(1, 20)]
    for att in all_attacks:
        att_items = [it for it in results if it["attack_id"] == att]
        if att_items:
            rec = np.mean([1 if it["is_correct"] else 0 for it in att_items]) * 100.0
            p_s = np.mean([it["p_spoof"] for it in att_items])
            cs = np.mean([it["consistency_score"] for it in att_items])
            d_p = np.mean([it["mean_abs_delta_p"] for it in att_items])
            attack_summary[att] = {
                "category": "Known" if att in [f"A{i:02d}" for i in range(1, 7)] else "Unseen",
                "count": len(att_items),
                "recall": round(float(rec), 2),
                "mean_p_spoof": round(float(p_s), 4),
                "mean_consistency_score": round(float(cs), 4),
                "mean_abs_delta_p": round(float(d_p), 4),
            }

    # Thống kê per-challenge impact
    challenge_names = ["gain_0.90", "gain_1.10", "noise_snr_35db", "resampling_15200hz"]
    challenge_impact = {}
    for ch_name in challenge_names:
        ch_delta_p = []
        ch_ev_delta = []
        flips_count = 0
        total_eval = 0
        for it in results:
            if ch_name in it["challenge_deltas"]:
                d = it["challenge_deltas"][ch_name]
                ch_delta_p.append(abs(d["delta_p"]))
                ch_ev_delta.append(d["evidence_delta"])
                orig_pred = it["prediction"]
                new_pred = "spoof" if d["p_spoof"] >= 0.5 else "bonafide"
                if orig_pred != new_pred:
                    flips_count += 1
                total_eval += 1

        challenge_impact[ch_name] = {
            "mean_abs_delta_p": round(float(np.mean(ch_delta_p)), 4) if ch_delta_p else 0.0,
            "std_abs_delta_p": round(float(np.std(ch_delta_p)), 4) if ch_delta_p else 0.0,
            "mean_evidence_delta": round(float(np.mean(ch_ev_delta)), 4) if ch_ev_delta else 0.0,
            "flip_count": flips_count,
            "flip_rate": round(flips_count / max(1, total_eval), 4),
        }

    return {
        "groups": group_summary,
        "attacks": attack_summary,
        "challenges": challenge_impact,
    }


def plot_figures(results: List[Dict[str, Any]], metrics: Dict[str, Any], figures_dir: Path):
    """Vẽ và lưu 5 biểu đồ báo cáo vào thư mục docs/figures/."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # Phân loại 3 tập chính: Bonafide (tất cả), Known (A01-A06), Unseen (A07-A19)
    bonafide_items = [r for r in results if r["label"] == "bonafide"]
    known_items = [r for r in results if r["group"] == "Known Attacks (A01-A06)"]
    unseen_items = [r for r in results if r["group"] == "Unseen Attacks (A07-A19)"]

    # -------------------------------------------------------------
    # Hình 1: docs/figures/p_spoof_distribution.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    data_p_spoof = [
        [r["p_spoof"] for r in bonafide_items],
        [r["p_spoof"] for r in known_items],
        [r["p_spoof"] for r in unseen_items],
    ]
    labels = ["Bonafide\n(Real Audio)", "Known Attacks\n(A01-A06)", "Unseen Attacks\n(A07-A19)"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    bp = ax.boxplot(data_p_spoof, tick_labels=labels, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Threshold (0.50)")
    ax.set_ylabel("P(spoof) — Classifier Output")
    ax.set_title("Figure 1: Distribution of P(spoof) across Audio Categories (Step 4 Audit)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig1_path = figures_dir / "p_spoof_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  -> Đã lưu Figure 1: {fig1_path}")

    # -------------------------------------------------------------
    # Hình 2: docs/figures/consistency_score_distribution.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    data_cs = [
        [r["consistency_score"] for r in bonafide_items],
        [r["consistency_score"] for r in known_items],
        [r["consistency_score"] for r in unseen_items],
    ]
    bp = ax.boxplot(data_cs, tick_labels=labels, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Acoustic Consistency Score [0, 1]")
    ax.set_title("Figure 2: Stability under Perturbations across Audio Categories")
    ax.set_ylim(0.5, 1.02)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig2_path = figures_dir / "consistency_score_distribution.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  -> Đã lưu Figure 2: {fig2_path}")

    # -------------------------------------------------------------
    # Hình 3: docs/figures/known_vs_unseen_comparison.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    known_recall = np.mean([1 if r["is_correct"] else 0 for r in known_items]) * 100
    unseen_recall = np.mean([1 if r["is_correct"] else 0 for r in unseen_items]) * 100
    known_mean_p = np.mean([r["p_spoof"] for r in known_items]) * 100
    unseen_mean_p = np.mean([r["p_spoof"] for r in unseen_items]) * 100
    known_cs = np.mean([r["consistency_score"] for r in known_items]) * 100
    unseen_cs = np.mean([r["consistency_score"] for r in unseen_items]) * 100

    metrics_names = ["Recall (%)", "Mean P(spoof) (%)", "Consistency Score (%)"]
    known_vals = [known_recall, known_mean_p, known_cs]
    unseen_vals = [unseen_recall, unseen_mean_p, unseen_cs]

    x = np.arange(len(metrics_names))
    width = 0.35

    rects1 = ax.bar(x - width/2, known_vals, width, label="Known Attacks (A01-A06)", color="#3498db", alpha=0.85)
    rects2 = ax.bar(x + width/2, unseen_vals, width, label="Unseen Attacks (A07-A19)", color="#e74c3c", alpha=0.85)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Figure 3: Known vs Unseen Spoofing Attacks Performance & Stability")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Thêm nhãn số trên đầu cột
    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h), ha="center", va="bottom", xytext=(0, 2), textcoords="offset points", fontsize=9)
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h), ha="center", va="bottom", xytext=(0, 2), textcoords="offset points", fontsize=9)

    fig3_path = figures_dir / "known_vs_unseen_comparison.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  -> Đã lưu Figure 3: {fig3_path}")

    # -------------------------------------------------------------
    # Hình 4: docs/figures/per_attack_recall.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    attacks = sorted(list(metrics["attacks"].keys()))
    recalls = [metrics["attacks"][a]["recall"] for a in attacks]
    bar_colors = ["#3498db" if metrics["attacks"][a]["category"] == "Known" else "#e74c3c" for a in attacks]

    bars = ax.bar(attacks, recalls, color=bar_colors, alpha=0.85)
    ax.axhline(80, color="gray", linestyle="--", alpha=0.5, label="80% Baseline Target")
    ax.set_ylabel("Detection Recall (%)")
    ax.set_xlabel("ASVspoof 2019 Attack Algorithm (A01-A06: Known, A07-A19: Unseen)")
    ax.set_title("Figure 4: Detection Recall per Attack Algorithm (A01 - A19)")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Thêm chú thích màu
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="Known Attacks (Train/Dev A01-A06)"),
        Patch(facecolor="#e74c3c", label="Unseen Attacks (Eval A07-A19)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}%", (bar.get_x() + bar.get_width()/2, h), ha="center", va="bottom", xytext=(0, 2), textcoords="offset points", fontsize=8)

    fig4_path = figures_dir / "per_attack_recall.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"  -> Đã lưu Figure 4: {fig4_path}")

    # -------------------------------------------------------------
    # Hình 5: docs/figures/challenge_delta_p_spoof.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ch_names = list(metrics["challenges"].keys())
    ch_display_names = ["Gain 0.90\n(-0.92 dB)", "Gain 1.10\n(+0.83 dB)", "Additive Noise\n(SNR 35 dB)", "Resampling\n(16k->15.2k->16k)"]
    mean_deltas = [metrics["challenges"][c]["mean_abs_delta_p"] for c in ch_names]
    ev_deltas = [metrics["challenges"][c]["mean_evidence_delta"] for c in ch_names]

    x = np.arange(len(ch_names))
    width = 0.35

    r1 = ax.bar(x - width/2, mean_deltas, width, label="Mean |Δ P(spoof)|", color="#8e44ad", alpha=0.85)
    r2 = ax.bar(x + width/2, ev_deltas, width, label="Mean Relative Feature Δ", color="#f39c12", alpha=0.85)

    ax.set_ylabel("Mean Perturbation Impact")
    ax.set_title("Figure 5: Acoustic Challenge Impact by Perturbation Type")
    ax.set_xticks(x)
    ax.set_xticklabels(ch_display_names)
    
    max_mean = max(mean_deltas) if mean_deltas else 0
    max_ev = max(ev_deltas) if ev_deltas else 0
    max_y = max(max_mean, max_ev)
    ax.set_ylim(0, max_y * 1.35 if max_y > 0 else 1.0)
    
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for r in r1:
        h = r.get_height()
        ax.annotate(f"{h:.4f}", (r.get_x() + r.get_width()/2, h), ha="center", va="bottom", xytext=(0, 2), textcoords="offset points", fontsize=8)
    for r in r2:
        h = r.get_height()
        ax.annotate(f"{h:.4f}", (r.get_x() + r.get_width()/2, h), ha="center", va="bottom", xytext=(0, 2), textcoords="offset points", fontsize=8)

    fig5_path = figures_dir / "challenge_delta_p_spoof.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"  -> Đã lưu Figure 5: {fig5_path}")


def print_formatted_summary(metrics: Dict[str, Any]):
    """In bảng tổng hợp kết quả chi tiết ra terminal."""
    print("\n" + "=" * 76)
    print("        BƯỚC 4 — BÁO CÁO ĐÁNH GIÁ CHUỖI CHỨNG CỨ ÂM HỌC & THỬ THÁCH")
    print("=" * 76)

    print("\n[BẢNG 1: SO SÁNH CÁC NHÓM ÂM THANH (GROUPS SUMMARY)]")
    print(f"{'Nhóm Âm Thanh':<28} | {'Số lượng':<8} | {'Accuracy/Recall':<15} | {'Mean P(spoof)':<14} | {'Consistency':<12}")
    print("-" * 76)
    for g_name, g_info in metrics["groups"].items():
        print(f"{g_name:<28} | {g_info['count']:<8} | {g_info['accuracy_or_recall']:>13.2f}% | {g_info['mean_p_spoof']:>14.4f} | {g_info['mean_consistency_score']:>11.4f}")

    print("\n[BẢNG 2: TÁC ĐỘNG CỦA CÁC THỬ THÁCH ÂM HỌC CÓ KIỂM SOÁT (CHALLENGES IMPACT)]")
    print(f"{'Thử Thách (Challenge)':<24} | {'Mean |Δ P(spoof)|':<18} | {'Mean Feature Δ':<16} | {'Flip Rate':<10}")
    print("-" * 76)
    for ch_name, ch_info in metrics["challenges"].items():
        print(f"{ch_name:<24} | {ch_info['mean_abs_delta_p']:>18.4f} | {ch_info['mean_evidence_delta']:>16.4f} | {ch_info['flip_rate']*100:>8.2f}%")

    print("\n[BẢNG 3: CHI TIẾT THEO THUẬT TOÁN TẤN CÔNG (A01 — A19)]")
    print(f"{'Attack':<8} | {'Loại':<8} | {'Số mẫu':<8} | {'Recall':<10} | {'Mean P(spoof)':<14} | {'Consistency':<12}")
    print("-" * 76)
    for att, info in metrics["attacks"].items():
        print(f"{att:<8} | {info['category']:<8} | {info['count']:<8} | {info['recall']:>8.2f}% | {info['mean_p_spoof']:>14.4f} | {info['mean_consistency_score']:>11.4f}")

    print("=" * 76 + "\n")


def main():
    args = parse_args()
    db_path = Path(args.db)
    model_path = Path(args.model)
    figures_dir = Path(args.figures_dir)
    metrics_out = Path(args.metrics_output)

    if not db_path.exists():
        print(f"Lỗi: Không tìm thấy database tại: {db_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"Lỗi: Không tìm thấy mô hình tại: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Khởi tạo AcousticAnalyzer từ model: {model_path} ...")
    analyzer = AcousticAnalyzer(model_path=model_path)

    print(f"Truy vấn lấy tập mẫu đánh giá từ DB ({args.samples_per_attack} mẫu/attack, {args.bonafide_samples} mẫu bonafide/split)...")
    samples = load_dataset_samples(
        db_path=db_path,
        samples_per_attack=args.samples_per_attack,
        bonafide_samples=args.bonafide_samples,
        seed=args.seed,
    )
    print(f"Đã chọn {len(samples)} mẫu hợp lệ.")

    results = run_batch_evaluation(analyzer, samples)
    metrics = compute_group_metrics(results)

    print_formatted_summary(metrics)

    print(f"Đang tạo các biểu đồ trực quan tại {figures_dir} ...")
    plot_figures(results, metrics, figures_dir)

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu báo cáo số liệu vào: {metrics_out.resolve()}")


if __name__ == "__main__":
    main()
