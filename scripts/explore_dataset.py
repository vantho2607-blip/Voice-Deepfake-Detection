#!/usr/bin/env python3
"""
scripts/explore_dataset.py

Khám phá và kiểm tra cấu trúc dataset ASVspoof 2019 LA:
- Kiểm tra thư mục dataset và các thư mục con
- Kiểm tra các protocol files (train, dev, eval)
- Đếm số records và hiển thị dòng mẫu
- Thống kê nhãn (bonafide vs spoof) và attack systems (A01-A19)
- Thống kê số lượng audio (.flac) và kích thước dataset

Chỉ sử dụng thư viện chuẩn Python (standard library).
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

# Đảm bảo UTF-8 output trên Windows console
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "data" / "LA" / "LA"
    if (PROJECT_ROOT / "data" / "LA" / "LA").exists()
    else PROJECT_ROOT / "data" / "archive" / "LA" / "LA"
)


def format_size(bytes_val: int) -> str:
    """Đổi bytes sang định dạng dễ đọc (MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def explore_dataset(data_root_str: str) -> None:
    data_root = Path(data_root_str)
    print("=" * 80)
    print(" BÁO CÁO KHÁM PHÁ DATASET: ASVspoof 2019 LA (Logical Access)")
    print("=" * 80)
    print(f"[*] Thư mục gốc dataset : {data_root}")

    if not data_root.exists():
        print(f"[!] LỖI: Thư mục dataset không tồn tại: {data_root}")
        sys.exit(1)

    # 1. Khám phá các thư mục con
    print("\n" + "-" * 80)
    print("1. CẤU TRÚC THƯ MỤC CHÍNH")
    print("-" * 80)
    subdirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    subfiles = sorted([f for f in data_root.iterdir() if f.is_file()])

    for d in subdirs:
        print(f"  [DIR]  {d.name}")
    for f in subfiles:
        print(f"  [FILE] {f.name} ({format_size(f.stat().st_size)})")

    # 2. Kiểm tra protocol files (Countermeasures - CM)
    cm_dir = data_root / "ASVspoof2019_LA_cm_protocols"
    if not cm_dir.exists():
        print(f"[!] LỖI: Không tìm thấy thư mục CM protocols tại {cm_dir}")
        return

    print("\n" + "-" * 80)
    print("2. PHÂN TÍCH PROTOCOL COUNTERMEASURES (CM)")
    print("-" * 80)

    protocol_files = [
        ("Train", cm_dir / "ASVspoof2019.LA.cm.train.trn.txt", data_root / "ASVspoof2019_LA_train" / "flac"),
        ("Dev", cm_dir / "ASVspoof2019.LA.cm.dev.trl.txt", data_root / "ASVspoof2019_LA_dev" / "flac"),
        ("Eval", cm_dir / "ASVspoof2019.LA.cm.eval.trl.txt", data_root / "ASVspoof2019_LA_eval" / "flac"),
    ]

    total_records_all = 0
    total_bonafide_all = 0
    total_spoof_all = 0

    for split_name, prot_file, flac_dir in protocol_files:
        print(f"\n>>> Split: {split_name.upper()} | File: {prot_file.name}")
        if not prot_file.exists():
            print(f"    [!] File không tồn tại: {prot_file}")
            continue

        sample_lines = []
        speakers = set()
        labels_count = Counter()
        systems_count = Counter()
        record_count = 0

        with open(prot_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line_str = line.strip()
                if not line_str:
                    continue
                parts = line_str.split()
                if len(parts) < 5:
                    continue
                record_count += 1
                spk, fname, _, sys_id, key = parts[:5]
                speakers.add(spk)
                labels_count[key] += 1
                systems_count[sys_id] += 1

                if line_idx < 5:
                    sample_lines.append(line_str)

        total_records_all += record_count
        total_bonafide_all += labels_count.get("bonafide", 0)
        total_spoof_all += labels_count.get("spoof", 0)

        # Đếm file flac tương ứng
        flac_count = len(list(flac_dir.glob("*.flac"))) if flac_dir.exists() else 0

        print(f"    - Tổng số records trong protocol : {record_count:,}")
        print(f"    - Số file audio .flac trên đĩa   : {flac_count:,}")
        print(f"    - Số speakers tham gia           : {len(speakers)}")
        print(f"    - Phân bố nhãn                   : Bonafide = {labels_count.get('bonafide', 0):,} | Spoof = {labels_count.get('spoof', 0):,}")
        
        # Thống kê attack systems
        spoof_systems = sorted([k for k in systems_count.keys() if k != "-"])
        sys_summary = ", ".join([f"{k}: {systems_count[k]:,}" for k in spoof_systems])
        print(f"    - Các hệ thống tấn công spoof     : {sys_summary if sys_summary else 'None'}")

        print("    - Mẫu 5 dòng đầu tiên:")
        for s in sample_lines:
            print(f"        {s}")

    # 3. Tổng hợp toàn bộ dataset
    print("\n" + "-" * 80)
    print("3. TỔNG HỢP TOÀN BỘ DATASET")
    print("-" * 80)
    print(f"  * Tổng số records trong toàn bộ CM protocol : {total_records_all:,}")
    if total_records_all > 0:
        print(f"  * Tổng bonafide                             : {total_bonafide_all:,} ({total_bonafide_all/total_records_all*100:.2f}%)")
        print(f"  * Tổng spoof                                : {total_spoof_all:,} ({total_spoof_all/total_records_all*100:.2f}%)")
    else:
        print(f"  * Tổng bonafide                             : 0 (0.00%)")
        print(f"  * Tổng spoof                                : 0 (0.00%)")
    print(f"  * Định dạng âm thanh                        : FLAC (16 kHz, 16-bit PCM mono)")

    # 4. Kích thước tổng quan
    print("\n" + "-" * 80)
    print("4. DUNG LƯỢNG LƯU TRỮ TRÊN ĐĨA")
    print("-" * 80)
    total_bytes = sum(f.stat().st_size for f in data_root.rglob("*") if f.is_file())
    total_audio_files = len(list(data_root.rglob("*.flac")))
    print(f"  * Tổng dung lượng thư mục : {format_size(total_bytes)}")
    print(f"  * Tổng số file .flac      : {total_audio_files:,}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Khám phá dataset ASVspoof 2019 LA")
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help=f"Đường dẫn thư mục dataset ASVspoof 2019 LA (mặc định: {DEFAULT_DATA_ROOT})",
    )
    args = parser.parse_args()
    explore_dataset(args.data_root)


if __name__ == "__main__":
    main()
