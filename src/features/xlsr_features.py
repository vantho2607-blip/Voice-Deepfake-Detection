"""
src/features/xlsr_features.py

Trích xuất đặc trưng sâu 1024 chiều sử dụng mô hình facebook/wav2vec2-xls-r-300m
"""

import warnings
import numpy as np

try:
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
except ImportError:
    warnings.warn("Thư viện torch hoặc transformers chưa được cài đặt. XlsrFeatureExtractor sẽ không hoạt động.")

class XlsrFeatureExtractor:
    """
    Bộ trích xuất đặc trưng dùng pre-trained XLS-R (1024 chiều).
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-xls-r-300m", device: str = None):
        self.model_name = model_name
        
        # Tự động chọn thiết bị (CUDA/MPS/CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[XLS-R] Đang khởi tạo mô hình {model_name} trên thiết bị {self.device}...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # XLS-R 300M có hidden_size là 1024
        self.feature_dim = self.model.config.hidden_size
        self.feature_names = [f"xlsr_{i}" for i in range(self.feature_dim)]

    def extract(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Trích xuất vector đặc trưng 1024 chiều bằng cách Mean Pooling hidden states.
        
        Args:
            waveform (np.ndarray): Audio waveform (1D array)
            sr (int): Sample rate (phải là 16000 đối với XLS-R)
            
        Returns:
            np.ndarray: Vector đặc trưng kích thước (1024,)
        """
        if sr != 16000:
            raise ValueError(f"XLS-R yêu cầu tần số lấy mẫu 16000Hz, nhận được {sr}Hz.")
            
        if len(waveform) == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Xử lý input với processor
        inputs = self.processor(
            waveform, 
            sampling_rate=sr, 
            return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        # Chạy qua mô hình
        with torch.no_grad():
            outputs = self.model(input_values)
            # Lấy hidden state cuối cùng (batch_size, sequence_length, hidden_size)
            last_hidden_state = outputs.last_hidden_state
            
        # Mean pooling theo chiều thời gian (sequence_length)
        pooled_features = torch.mean(last_hidden_state, dim=1)
            
        # Chuyển về numpy array 1D
        feature_vector = pooled_features.squeeze(0).cpu().numpy().astype(np.float32)
        
        return feature_vector
