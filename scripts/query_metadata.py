#!/usr/bin/env python3
"""
scripts/query_metadata.py

Demo truy vấn SQL trên SQLite metadata database (data/processed/metadata.db):
1. Tổng số sample
2. Số lượng bonafide / spoof (kèm tỉ lệ %)
3. Phân bố theo attack type (A01 - A19)
4. Phân bố theo speaker
5. Phân bố theo train / dev / eval
6. Query JOIN (bảng samples JOIN attack_types) phân tích theo kỹ thuật sinh giọng (TTS vs VC)
7. Query GROUP BY nâng cao (ma trận Attack System theo từng Split để kiểm tra known vs unseen attacks)

Chỉ sử dụng thư viện chuẩn Python (sqlite3, pathlib, argparse, sys).
"""

import argparse
from pathlib import Path
import sqlite3
import sys

# Đảm bảo UTF-8 output trên Windows console
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.db"


def print_table(headers: list[str], rows: list[tuple], title: str, sql_query: str) -> None:
    """In kết quả truy vấn dạng bảng ASCII đẹp mắt."""
    print("\n" + "=" * 90)
    print(f" {title.upper()}")
    print("=" * 90)
    print("SQL Query:")
    for line in sql_query.strip().split("\n"):
        print(f"  {line.strip()}")
    print("-" * 90)

    if not rows:
        print("  (Không có dữ liệu)")
        return

    # Tính độ rộng từng cột
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            val_str = str(val if val is not None else "NULL")
            if len(val_str) > col_widths[idx]:
                col_widths[idx] = len(val_str)

    # Padding thêm 2 spaces
    col_widths = [w + 2 for w in col_widths]

    # Format header
    header_str = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    sep_str = "".join("-" * (w - 1) + " " for w in col_widths)

    print(header_str)
    print(sep_str)

    # Format rows
    for row in rows:
        row_str = ""
        for idx, val in enumerate(row):
            val_str = str(val if val is not None else "NULL")
            # Căn phải nếu là số, căn trái nếu là chuỗi
            if isinstance(val, (int, float)):
                if isinstance(val, int):
                    val_str = f"{val:,}"
                row_str += f"{val_str:>{col_widths[idx]-2}}  "
            else:
                row_str += f"{val_str:<{col_widths[idx]}}"
        print(row_str)

    print(f"\n[Tổng số dòng kết quả: {len(rows)}]")


def run_queries(db_path_str: str) -> None:
    db_path = Path(db_path_str)
    if not db_path.exists():
        print(f"[!] LỖI: Database không tồn tại: {db_path}")
        print("    Vui lòng chạy 'python scripts/build_metadata_db.py' trước.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # -------------------------------------------------------------
    # QUERY 1: TỔNG SỐ SAMPLE
    # -------------------------------------------------------------
    sql_1 = "SELECT COUNT(*) AS total_samples FROM samples;"
    cursor.execute(sql_1)
    rows_1 = cursor.fetchall()
    print_table(["Total Samples"], rows_1, "1. Tổng số mẫu (Samples)", sql_1)

    # -------------------------------------------------------------
    # QUERY 2: SỐ LƯỢNG BONAFIDE / SPOOF
    # -------------------------------------------------------------
    sql_2 = """
    SELECT 
        label,
        COUNT(*) AS count,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM samples), 2) AS percentage
    FROM samples
    GROUP BY label;
    """
    cursor.execute(sql_2)
    rows_2 = cursor.fetchall()
    print_table(["Label", "Count", "Percentage (%)"], rows_2, "2. Phân bố Bonafide vs Spoof", sql_2)

    # -------------------------------------------------------------
    # QUERY 3: PHÂN BỐ THEO ATTACK TYPE
    # -------------------------------------------------------------
    sql_3 = """
    SELECT 
        COALESCE(attack_id, 'None (Bonafide)') AS attack_type,
        COUNT(*) AS count,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM samples), 2) AS percentage
    FROM samples
    GROUP BY attack_id
    ORDER BY 
        CASE WHEN attack_id IS NULL THEN 0 ELSE 1 END,
        attack_id;
    """
    cursor.execute(sql_3)
    rows_3 = cursor.fetchall()
    print_table(["Attack Type", "Count", "Percentage (%)"], rows_3, "3. Phân bố theo Attack Type (A01 - A19)", sql_3)

    # -------------------------------------------------------------
    # QUERY 4: PHÂN BỐ THEO SPEAKER
    # -------------------------------------------------------------
    sql_4 = """
    SELECT 
        speaker_id,
        split,
        COUNT(*) AS total_utterances,
        SUM(CASE WHEN label = 'bonafide' THEN 1 ELSE 0 END) AS bonafide_count,
        SUM(CASE WHEN label = 'spoof' THEN 1 ELSE 0 END) AS spoof_count
    FROM samples
    GROUP BY speaker_id, split
    ORDER BY total_utterances DESC
    LIMIT 10;
    """
    cursor.execute(sql_4)
    rows_4 = cursor.fetchall()
    print_table(
        ["Speaker ID", "Split", "Total Utterances", "Bonafide Count", "Spoof Count"],
        rows_4,
        "4. Phân bố theo Speaker (Top 10 Speakers có nhiều utterance nhất)",
        sql_4,
    )

    # -------------------------------------------------------------
    # QUERY 5: PHÂN BỐ THEO TRAIN / DEV / EVAL
    # -------------------------------------------------------------
    sql_5 = """
    SELECT 
        split,
        COUNT(DISTINCT speaker_id) AS num_speakers,
        COUNT(*) AS total_samples,
        SUM(CASE WHEN label = 'bonafide' THEN 1 ELSE 0 END) AS bonafide_count,
        SUM(CASE WHEN label = 'spoof' THEN 1 ELSE 0 END) AS spoof_count,
        ROUND(SUM(CASE WHEN label = 'bonafide' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS bonafide_pct,
        ROUND(SUM(CASE WHEN label = 'spoof' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS spoof_pct
    FROM samples
    GROUP BY split
    ORDER BY 
        CASE split 
            WHEN 'train' THEN 1 
            WHEN 'dev' THEN 2 
            WHEN 'eval' THEN 3 
        END;
    """
    cursor.execute(sql_5)
    rows_5 = cursor.fetchall()
    print_table(
        ["Split", "Speakers", "Total Samples", "Bonafide", "Spoof", "Bonafide %", "Spoof %"],
        rows_5,
        "5. Phân bố theo Dataset Split (Train / Dev / Eval)",
        sql_5,
    )

    # -------------------------------------------------------------
    # QUERY 6: QUERY JOIN (samples JOIN attack_types)
    # -------------------------------------------------------------
    sql_6 = """
    SELECT 
        a.method AS synthesis_method,
        COUNT(DISTINCT a.attack_id) AS attack_systems_count,
        COUNT(s.id) AS total_spoof_samples,
        ROUND(COUNT(s.id) * 100.0 / (SELECT COUNT(*) FROM samples WHERE label = 'spoof'), 2) AS pct_of_all_spoofs
    FROM samples s
    INNER JOIN attack_types a ON s.attack_id = a.attack_id
    GROUP BY a.method
    ORDER BY total_spoof_samples DESC;
    """
    cursor.execute(sql_6)
    rows_6 = cursor.fetchall()
    print_table(
        ["Synthesis Method", "Num Attack Systems", "Total Spoof Samples", "% of All Spoofs"],
        rows_6,
        "6. JOIN QUERY: Thống kê Spoof theo phương pháp tổng hợp (TTS vs VC vs TTS_VC)",
        sql_6,
    )

    # -------------------------------------------------------------
    # QUERY 7: QUERY GROUP BY NÂNG CAO (Ma trận Known vs Unseen Attacks)
    # -------------------------------------------------------------
    sql_7 = """
    SELECT 
        COALESCE(s.attack_id, 'Bonafide') AS attack,
        COALESCE(a.method, 'Human') AS method,
        COALESCE(a.description, 'Natural Speech') AS description,
        SUM(CASE WHEN s.split = 'train' THEN 1 ELSE 0 END) AS train_count,
        SUM(CASE WHEN s.split = 'dev' THEN 1 ELSE 0 END) AS dev_count,
        SUM(CASE WHEN s.split = 'eval' THEN 1 ELSE 0 END) AS eval_count,
        COUNT(*) AS total_count
    FROM samples s
    LEFT JOIN attack_types a ON s.attack_id = a.attack_id
    GROUP BY s.attack_id, a.method, a.description
    ORDER BY 
        CASE WHEN s.attack_id IS NULL THEN 0 ELSE 1 END,
        s.attack_id;
    """
    cursor.execute(sql_7)
    rows_7 = cursor.fetchall()
    print_table(
        ["Attack", "Method", "Algorithm Description", "Train", "Dev", "Eval", "Total"],
        rows_7,
        "7. GROUP BY QUERY: Ma trận phân bố Attack System qua các Split (Known vs Unseen)",
        sql_7,
    )

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Demo các truy vấn SQL trên SQLite metadata database")
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Đường dẫn SQLite database (mặc định: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()
    run_queries(args.db_path)


if __name__ == "__main__":
    main()
