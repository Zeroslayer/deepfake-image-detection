import gradio as gr
import torch
import cv2
import numpy as np
from predict_deepfake import load_model, val_transform, DeepfakeDetector

# Load model 1 lan khi khoi chay app
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('weights/best_model.pth', device)

def predict(image):
    if image is None:
        return "Vui long tải ảnh lên", 0.0, 0.0
    
    # Chuyen anh PIL sang numpy/RGB
    img = np.array(image)
    tensor = val_transform(image=img)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    real_prob = float(probs[0])
    fake_prob = float(probs[1])

    # Tra ve ket qua duoi dang dict cho Gradio hien thi thanh phan pho
    return {"Real": real_prob, "Fake": fake_prob}

# Tao giao dien Web
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Tải ảnh cần kiểm tra"),
    outputs=gr.Label(num_top_classes=2, label="Kết quả dự đoán"),
    title="Deepfake Image Detection",
    description="Tải lên bức ảnh khuôn mặt để kiểm tra xem là ảnh Thật (Real) hay Giả mạo (Fake)."
)

if __name__ == "__main__":
    interface.launch()
