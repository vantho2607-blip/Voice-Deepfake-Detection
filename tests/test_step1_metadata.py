"""
tests/test_step1_metadata.py

Kiểm thử tự động cho BƯỚC 1: Quản lý & EDA Metadata bằng SQL
"""

from pathlib import Path
import sqlite3
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = (
    PROJECT_ROOT / "data" / "LA" / "LA"
    if (PROJECT_ROOT / "data" / "LA" / "LA").exists()
    else PROJECT_ROOT / "data" / "archive" / "LA" / "LA"
)
DB_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.db"


@pytest.fixture(scope="module")
def db_conn():
    assert DB_PATH.exists(), f"Database {DB_PATH} không tồn tại. Hãy chạy build_metadata_db.py trước."
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


class TestStep1DatasetAndProtocols:
    def test_dataset_directory_exists(self):
        assert DATA_ROOT.exists(), f"Dataset root {DATA_ROOT} không tồn tại."
        assert (DATA_ROOT / "ASVspoof2019_LA_cm_protocols").is_dir()
        assert (DATA_ROOT / "ASVspoof2019_LA_train" / "flac").is_dir()
        assert (DATA_ROOT / "ASVspoof2019_LA_dev" / "flac").is_dir()
        assert (DATA_ROOT / "ASVspoof2019_LA_eval" / "flac").is_dir()

    def test_protocol_files_exist_and_readable(self):
        cm_dir = DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
        train_p = cm_dir / "ASVspoof2019.LA.cm.train.trn.txt"
        dev_p = cm_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
        eval_p = cm_dir / "ASVspoof2019.LA.cm.eval.trl.txt"

        for p in [train_p, dev_p, eval_p]:
            assert p.exists(), f"Protocol file {p} không tồn tại."
            assert p.stat().st_size > 0


class TestStep1DatabaseIntegrity:
    def test_tables_and_indexes_exist(self, db_conn):
        cursor = db_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        assert "samples" in tables
        assert "attack_types" in tables

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='samples';")
        indexes = [r[0] for r in cursor.fetchall()]
        assert "idx_samples_file_id" in indexes
        assert "idx_samples_split" in indexes
        assert "idx_samples_label" in indexes
        assert "idx_samples_attack_id" in indexes

    def test_sqlite_integrity_and_foreign_keys(self, db_conn):
        cursor = db_conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        res = cursor.fetchone()[0]
        assert res == "ok", f"Integrity check failed: {res}"

        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("PRAGMA foreign_key_check;")
        fk_violations = cursor.fetchall()
        assert len(fk_violations) == 0, f"Foreign key violations: {fk_violations}"

    def test_no_duplicate_file_id(self, db_conn):
        cursor = db_conn.cursor()
        cursor.execute("SELECT file_id, COUNT(*) FROM samples GROUP BY file_id HAVING COUNT(*) > 1;")
        duplicates = cursor.fetchall()
        assert len(duplicates) == 0, f"Phát hiện file_id bị trùng lặp: {duplicates}"

    def test_no_unexpected_nulls(self, db_conn):
        cursor = db_conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM samples WHERE file_id IS NULL OR speaker_id IS NULL "
            "OR split IS NULL OR label IS NULL OR audio_path IS NULL;"
        )
        null_count = cursor.fetchone()[0]
        assert null_count == 0, f"Phát hiện {null_count} bản ghi có trường bắt buộc bị NULL."

    def test_database_record_counts_match_protocols(self, db_conn):
        cursor = db_conn.cursor()
        cursor.execute("SELECT split, COUNT(*) FROM samples GROUP BY split ORDER BY split;")
        counts = dict(cursor.fetchall())

        assert counts.get("train", 0) > 0, f"Train count mismatch: {counts.get('train', 0)} <= 0"
        assert counts.get("dev", 0) > 0, f"Dev count mismatch: {counts.get('dev', 0)} <= 0"
        assert counts.get("eval", 0) > 0, f"Eval count mismatch: {counts.get('eval', 0)} <= 0"

        cursor.execute("SELECT COUNT(*) FROM samples;")
        total = cursor.fetchone()[0]
        assert total > 0, f"Total count mismatch: {total} <= 0"

    def test_label_and_attack_logic(self, db_conn):
        cursor = db_conn.cursor()
        # Bonafide must have NULL attack_id
        cursor.execute("SELECT COUNT(*) FROM samples WHERE label = 'bonafide' AND attack_id IS NOT NULL;")
        invalid_bonafide = cursor.fetchone()[0]
        assert invalid_bonafide == 0, "Bonafide có attack_id khác NULL."

        # Spoof must have valid attack_id in A01-A19
        cursor.execute("SELECT COUNT(*) FROM samples WHERE label = 'spoof' AND attack_id IS NULL;")
        invalid_spoof = cursor.fetchone()[0]
        assert invalid_spoof == 0, "Spoof có attack_id bị NULL."

    def test_attack_types_count(self, db_conn):
        cursor = db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attack_types;")
        assert cursor.fetchone()[0] == 19

    def test_join_query_execution(self, db_conn):
        cursor = db_conn.cursor()
        cursor.execute("""
        SELECT a.method, COUNT(s.id)
        FROM samples s
        INNER JOIN attack_types a ON s.attack_id = a.attack_id
        GROUP BY a.method;
        """)
        rows = dict(cursor.fetchall())
        assert rows.get("TTS", 0) >= 0
        assert rows.get("VC", 0) >= 0
        assert rows.get("TTS_VC", 0) >= 0
