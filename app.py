import streamlit as st
import torch
import numpy as np
from PIL import Image
from predict_deepfake import load_model, val_transform, DeepfakeDetector

st.title("🛡️ Deepfake Image Detection")
st.write("Tải ảnh lên để kiểm tra xem là ảnh **Real (Thật)** hay **Fake (Giả)**.")

@st.cache_resource
def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('weights/best_model.pth', device)
    return model, device

model, device = get_model()

uploaded_file = st.file_uploader("Chọn một bức ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    
    if st.button('Phân tích'):
        img_np = np.array(image)
        tensor = val_transform(image=img_np)['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        real_prob = float(probs[0]) * 100
        fake_prob = float(probs[1]) * 100

        st.subheader("Kết quả dự đoán:")
        if fake_prob >= 50:
            st.error(f"🚨 **FAKE (Giả mạo)** - Độ tin cậy: {fake_prob:.2f}%")
        else:
            st.success(f"✅ **REAL (Thật)** - Độ tin cậy: {real_prob:.2f}%")
            
        st.bar_chart({"Real (%)": real_prob, "Fake (%)": fake_prob})
