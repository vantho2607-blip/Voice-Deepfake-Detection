import sqlite3
import numpy as np
import torch
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class WavLMFeatureExtractor:
    def __init__(self, model_name="microsoft/wavlm-base-plus"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"[WavLM] Dang khoi tao mo hinh {model_name} tren {self.device}...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.feature_dim = self.model.config.hidden_size

    def extract(self, audio_path):
        waveform, sr = librosa.load(audio_path, sr=16000)
        
        # Tiền xử lý (chuẩn hóa RMS cơ bản)
        if len(waveform) > 0:
            rms = np.sqrt(np.mean(waveform**2))
            if rms > 0:
                waveform = waveform / rms * 0.05
                
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            last_hidden_state = outputs.last_hidden_state
            
        pooled_features = torch.mean(last_hidden_state, dim=1)
        return pooled_features.squeeze(0).cpu().numpy().astype(np.float32)

def extract_subset(db_path, split_name, extractor, limit=None):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    query = f"SELECT id, file_id, label, audio_path FROM samples WHERE split = '{split_name}'"
    if limit:
        query += f" ORDER BY RANDOM() LIMIT {limit}"
        
    c.execute(query)
    rows = c.fetchall()
    
    X = []
    y = []
    meta = []
    
    print(f"\n[+] Dang trich xuat WavLM cho tap {split_name.upper()} ({len(rows)} files)...")
    for row in tqdm(rows):
        sample_id, file_id, label, audio_path = row
        abs_path = PROJECT_ROOT / audio_path
        
        if abs_path.exists():
            feat = extractor.extract(str(abs_path))
            X.append(feat)
            y.append(0 if label == 'bonafide' else 1)
            meta.append({'id': sample_id, 'file_id': file_id, 'label': label})
            
    conn.close()
    return np.array(X), np.array(y), meta

def main():
    print("="*60)
    print("CHIEN DICH: THAY NAO HE THONG SANG WavLM-Base-Plus")
    print("="*60)
    
    db_path = PROJECT_ROOT / "data" / "processed" / "metadata.db"
    out_dir = PROJECT_ROOT / "data" / "processed" / "features_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = WavLMFeatureExtractor()
    
    # Không giới hạn nữa, chạy FULL toàn bộ Dataset!
    N_TRAIN = None
    N_EVAL = None
    
    print("\n[!] LUU Y: Chien dich nhai FULL Dataset bat dau. Se mat khoang 1 tieng!")
    
    # 1. Trích xuất Train
    X_tr, y_tr, m_tr = extract_subset(db_path, "train", extractor, limit=N_TRAIN)
    train_cache = out_dir / f"wavlm_base_train_{len(X_tr)}.npz"
    np.savez_compressed(train_cache, X=X_tr, y=y_tr, meta=m_tr)
    
    # 2. Trích xuất Eval
    X_ev, y_ev, m_ev = extract_subset(db_path, "eval", extractor, limit=N_EVAL)
    eval_cache = out_dir / f"wavlm_base_eval_{len(X_ev)}.npz"
    np.savez_compressed(eval_cache, X=X_ev, y=y_ev, meta=m_ev)
    
    print("\n[+] Da thay nao thanh cong full dataset! San sang mang di Benchmark!")

if __name__ == "__main__":
    main()
