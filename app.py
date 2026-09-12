import streamlit as st
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import time
import os
from src.features.audio_features import AudioFeatureExtractor
# Note: Để demo nhanh gọn không cần PyTorch, ta chỉ dùng bộ đặc trưng âm học vật lý (LFCC).
# Trong thực tế, bạn có thể load lại mô hình WavLM bằng cách gọi WavLMFeatureExtractor.

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Voice Deepfake Detector", page_icon="🎙️", layout="centered")

st.title("🎙️ Hệ thống Nhận diện Giọng nói Giả mạo")
st.markdown("**Dự án môn học DAP391m** | Phát hiện âm thanh do AI sinh ra (Deepfake/TTS)")

st.divider()

# --- KHỞI TẠO MÔ HÌNH ---
@st.cache_resource
def load_models():
    # Load mô hình Logistic Regression đã train (Bạn cần train và save model ra file .joblib trước)
    # Vì mục đích demo, nếu chưa có file, hệ thống sẽ báo lỗi nhẹ.
    model_path = "data/processed/lr_model.joblib"
    scaler_path = "data/processed/scaler.joblib"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

model, scaler = load_models()
extractor = AudioFeatureExtractor()

st.info("💡 **Hướng dẫn:** Hãy tải lên một đoạn ghi âm giọng nói (.wav, .flac) để hệ thống AI phân tích.")

# --- TẢI FILE LÊN ---
uploaded_file = st.file_uploader("Tải tệp âm thanh của bạn lên", type=['wav', 'flac', 'mp3'])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("🚀 Phân tích Âm thanh", use_container_width=True):
        if model is None:
            st.error("⚠️ Chưa tìm thấy mô hình đã huấn luyện (lr_model.joblib). Vui lòng huấn luyện mô hình trước!")
        else:
            with st.spinner("Đang xử lý tín hiệu (Resampling, VAD, Normalization)..."):
                time.sleep(1) # Fake delay cho chuyên nghiệp
                # Lưu file tạm để librosa đọc
                with open("temp_audio.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Trích xuất đặc trưng
                st.toast("Đang trích xuất đặc trưng không gian siêu chiều...")
                feat = extractor.extract_features("temp_audio.wav")
                
                # Phân loại
                X = scaler.transform([feat])
                prob_fake = model.predict_proba(X)[0][1]
                
                st.divider()
                st.subheader("📊 Kết quả Phân tích")
                
                if prob_fake > 0.5:
                    st.error(f"🚨 CẢNH BÁO: Đây là GIỌNG NÓI GIẢ MẠO (DEEPFAKE)!")
                    st.progress(float(prob_fake), text=f"Độ tự tin: {prob_fake*100:.2f}%")
                else:
                    st.success(f"✅ AN TOÀN: Đây là GIỌNG NÓI THẬT (BONAFIDE).")
                    st.progress(float(1 - prob_fake), text=f"Độ tự tin: {(1-prob_fake)*100:.2f}%")
                    
            os.remove("temp_audio.wav")
