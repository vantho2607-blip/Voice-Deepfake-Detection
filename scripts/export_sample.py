import sqlite3
import pandas as pd
from pathlib import Path

def export_sample():
    db_path = "data/processed/metadata.db"
    conn = sqlite3.connect(db_path)
    
    # Lấy 50 mẫu thật
    df_real = pd.read_sql_query("SELECT * FROM samples WHERE label='bonafide' ORDER BY RANDOM() LIMIT 50;", conn)
    
    # Lấy 150 mẫu giả
    df_spoof = pd.read_sql_query("SELECT * FROM samples WHERE label='spoof' ORDER BY RANDOM() LIMIT 150;", conn)
    
    # Nối và trộn đều
    df = pd.concat([df_real, df_spoof]).sample(frac=1, random_state=42)
    
    # Bỏ cột id tự tăng của DB nếu không cần thiết
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    out_path = Path("sample_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Đã xuất 200 dòng dữ liệu mẫu ra file: {out_path.absolute()}")

if __name__ == "__main__":
    export_sample()
