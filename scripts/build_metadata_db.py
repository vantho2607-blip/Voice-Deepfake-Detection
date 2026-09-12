#!/usr/bin/env python3
"""
scripts/build_metadata_db.py

Xây dựng SQLite database để quản lý metadata ASVspoof 2019 LA:
- Đọc các protocol Countermeasures (CM): train, dev, eval
- Phân tích và chuẩn hóa metadata: file_id, speaker_id, split, label, attack_id, audio_path
- Lưu trữ vào SQLite database (data/processed/metadata.db)
- Khởi tạo bảng attack_types với thông tin kỹ thuật từ README.LA.txt
- Tạo các chỉ mục (indexes) để tối ưu hóa truy vấn SQL
- Chạy lại an toàn không tạo bản ghi trùng lặp (idempotent)

Chỉ sử dụng thư viện chuẩn Python (sqlite3, pathlib, argparse, sys, time).
"""

import argparse
from pathlib import Path
import sqlite3
import sys
import time

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
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.db"

# Định nghĩa các hệ thống tấn công theo README.LA.txt
ATTACK_DEFINITIONS = [
    ("A01", "TTS", "Neural waveform model"),
    ("A02", "TTS", "Vocoder"),
    ("A03", "TTS", "Vocoder"),
    ("A04", "TTS", "Waveform concatenation"),
    ("A05", "VC", "Vocoder"),
    ("A06", "VC", "Spectral filtering"),
    ("A07", "TTS", "Vocoder + GAN"),
    ("A08", "TTS", "Neural waveform"),
    ("A09", "TTS", "Vocoder"),
    ("A10", "TTS", "Neural waveform"),
    ("A11", "TTS", "Griffin-Lim"),
    ("A12", "TTS", "Neural waveform"),
    ("A13", "TTS_VC", "Waveform concatenation + waveform filtering"),
    ("A14", "TTS_VC", "Vocoder"),
    ("A15", "TTS_VC", "Neural waveform"),
    ("A16", "TTS", "Waveform concatenation"),
    ("A17", "VC", "Waveform filtering"),
    ("A18", "VC", "Vocoder"),
    ("A19", "VC", "Spectral filtering"),
]


def init_database(conn: sqlite3.Connection, rebuild: bool = False) -> None:
    """Tạo schema cho SQLite database."""
    cursor = conn.cursor()

    if rebuild:
        print("[*] Rebuild mode: Xóa bảng cũ...")
        cursor.execute("DROP TABLE IF EXISTS samples;")
        cursor.execute("DROP TABLE IF EXISTS attack_types;")

    # 1. Bảng loại tấn công (attack_types)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attack_types (
        attack_id TEXT PRIMARY KEY,
        method TEXT NOT NULL,
        description TEXT NOT NULL
    );
    """)

    # 2. Bảng mẫu âm thanh (samples)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id TEXT UNIQUE NOT NULL,
        speaker_id TEXT NOT NULL,
        split TEXT NOT NULL,
        label TEXT NOT NULL,
        attack_id TEXT,
        protocol_file TEXT NOT NULL,
        audio_path TEXT NOT NULL,
        FOREIGN KEY (attack_id) REFERENCES attack_types (attack_id)
    );
    """)

    # 3. Tạo các chỉ mục (indexes) để tăng tốc độ truy vấn
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_file_id ON samples(file_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_split ON samples(split);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_attack_id ON samples(attack_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_speaker_id ON samples(speaker_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_split_label ON samples(split, label);")

    # 4. Chèn thông tin attack_types
    cursor.executemany("""
    INSERT OR REPLACE INTO attack_types (attack_id, method, description)
    VALUES (?, ?, ?);
    """, ATTACK_DEFINITIONS)

    conn.commit()


def parse_and_insert_split(
    conn: sqlite3.Connection,
    split_name: str,
    protocol_path: Path,
    flac_dir: Path,
) -> tuple[int, int]:
    """
    Parse file protocol và chèn records vào database.
    Trả về (inserted_count, skipped_count).
    """
    if not protocol_path.exists():
        print(f"[!] Cảnh báo: Protocol {protocol_path} không tồn tại.")
        return 0, 0

    cursor = conn.cursor()
    records = []

    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id, file_id, _, sys_id, label = parts[:5]
            attack_id = None if sys_id == "-" else sys_id
            audio_path = str(flac_dir / f"{file_id}.flac")

            records.append((
                file_id,
                speaker_id,
                split_name,
                label,
                attack_id,
                protocol_path.name,
                audio_path,
            ))

    cursor.execute("SELECT COUNT(*) FROM samples WHERE split = ?;", (split_name,))
    existing_in_split = cursor.fetchone()[0]

    # Dùng INSERT OR IGNORE để tránh trùng lặp nếu record đã có
    cursor.executemany("""
    INSERT OR IGNORE INTO samples (
        file_id, speaker_id, split, label, attack_id, protocol_file, audio_path
    ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """, records)

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM samples WHERE split = ?;", (split_name,))
    total_in_split = cursor.fetchone()[0]

    inserted = total_in_split - existing_in_split
    skipped = len(records) - inserted

    return inserted, skipped


def build_metadata_db(data_root_str: str, db_path_str: str, rebuild: bool = False) -> None:
    data_root = Path(data_root_str)
    db_path = Path(db_path_str)

    print("=" * 80)
    print(" XÂY DỰNG SQLITE METADATA DATABASE")
    print("=" * 80)
    print(f"[*] Dataset Root : {data_root}")
    print(f"[*] Database Path: {db_path}")

    if not data_root.exists():
        print(f"[!] LỖI: Dataset root không tồn tại: {data_root}")
        sys.exit(1)

    # Đảm bảo thư mục đích tồn tại
    db_path.parent.mkdir(parents=True, exist_ok=True)

    cm_dir = data_root / "ASVspoof2019_LA_cm_protocols"
    splits = [
        ("train", cm_dir / "ASVspoof2019.LA.cm.train.trn.txt", data_root / "ASVspoof2019_LA_train" / "flac"),
        ("dev", cm_dir / "ASVspoof2019.LA.cm.dev.trl.txt", data_root / "ASVspoof2019_LA_dev" / "flac"),
        ("eval", cm_dir / "ASVspoof2019.LA.cm.eval.trl.txt", data_root / "ASVspoof2019_LA_eval" / "flac"),
    ]

    start_time = time.time()
    conn = sqlite3.connect(db_path)

    # Khởi tạo tables và indexes
    init_database(conn, rebuild=rebuild)

    total_inserted = 0
    total_skipped = 0

    print("\n" + "-" * 80)
    print("TIẾN TRÌNH IMPORT DỮ LIỆU:")
    print("-" * 80)

    for split_name, prot_path, flac_dir in splits:
        inserted, skipped = parse_and_insert_split(conn, split_name, prot_path, flac_dir)
        total_inserted += inserted
        total_skipped += skipped
        print(f"  [+] Split '{split_name}': Đã insert mới {inserted:,} records (bỏ qua/trùng lặp: {skipped:,})")

    # Kiểm tra tổng số records trong DB
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM samples;")
    total_samples = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM attack_types;")
    total_attack_types = cursor.fetchone()[0]

    elapsed = time.time() - start_time
    conn.close()

    db_size_mb = db_path.stat().st_size / (1024 * 1024)

    print("\n" + "-" * 80)
    print("HOÀN THÀNH BUILD METADATA DATABASE:")
    print("-" * 80)
    print(f"  * Tổng số records trong bảng 'samples'     : {total_samples:,}")
    print(f"  * Tổng số attack types trong 'attack_types' : {total_attack_types}")
    print(f"  * Số bản ghi mới được insert                : {total_inserted:,}")
    print(f"  * Số bản ghi bỏ qua (đã tồn tại)           : {total_skipped:,}")
    print(f"  * Dung lượng file database                 : {db_size_mb:.2f} MB")
    print(f"  * Thời gian thực hiện                       : {elapsed:.2f} giây")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Tạo SQLite database từ metadata ASVspoof 2019 LA")
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help=f"Đường dẫn dataset ASVspoof 2019 LA (mặc định: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Đường dẫn SQLite database (mặc định: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Xóa database cũ và build lại từ đầu",
    )
    args = parser.parse_args()
    build_metadata_db(args.data_root, args.db_path, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
