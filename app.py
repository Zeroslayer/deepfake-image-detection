import os
import streamlit as st
import torch
import numpy as np
import gdown
from PIL import Image
from predict_deepfake import load_model, val_transform, DeepfakeDetector

st.title("🛡️ Deepfake Image Detection")
st.write("Tải ảnh lên để kiểm tra xem là ảnh **Real (Thật)** hay **Fake (Giả)**.")

MODEL_PATH = 'weights/best_model.pth'
# ĐÃ ĐIỀN ĐÚNG CHUỖI FILE ID CỦA BẠN:
GDRIVE_FILE_ID = '1n1BfdbCNzU5eaIzaHyQ9hqQXEp8qoVxh'

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs('weights', exist_ok=True)
        with st.spinner('Đang tải mô hình weights từ Google Drive (chỉ mất khoảng 10-15 giây lần đầu)...'):
            # Dùng tham số id=GDRIVE_FILE_ID trực tiếp
            gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, device)
    return model, device

uploaded_file = st.file_uploader("Kéo thả hoặc chọn bức ảnh cần kiểm tra...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
    
    if st.button('🔍 Phân tích ngay'):
        try:
            with st.spinner('Đang phân tích khuôn mặt...'):
                model, device = get_model()
                img_np = np.array(image)
                tensor = val_transform(image=img_np)['image'].unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]

                real_prob = float(probs[0]) * 100
                fake_prob = float(probs[1]) * 100

                st.subheader("📊 Kết quả dự đoán:")
                if fake_prob >= 50:
                    st.error(f"🚨 **FAKE (Ảnh Giả Mạo)** - Độ tin cậy: {fake_prob:.2f}%")
                else:
                    st.success(f"✅ **REAL (Ảnh Thật)** - Độ tin cậy: {real_prob:.2f}%")
                    
                st.bar_chart({"Real (%)": real_prob, "Fake (%)": fake_prob})
        except Exception as e:
            st.error(f"Xảy ra lỗi trong quá trình xử lý: {e}")
