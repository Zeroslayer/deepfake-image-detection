# deepfake-image-detection
AI-based Deepfake Image Detection using EfficientNet-B3 and PyTorch
## 📌 Introduction
Dự án này tập trung vào việc phân loại và phát hiện hình ảnh giả mạo (Deepfake) bằng các mô hình học sâu. Hệ thống nhận đầu vào là một bức ảnh và trả về kết quả dự đoán (Real hoặc Fake) kèm theo xác suất phân bố chi tiết, giúp người dùng dễ dàng đánh giá độ tin cậy của hình ảnh.

## 🎯 Features
* **Phân loại ảnh nhanh chóng:** Đánh giá ảnh đầu vào là Real hay Fake với độ trễ thấp.
* **Trực quan hóa kết quả:** Hiển thị ảnh gốc kèm theo biểu đồ thanh (bar chart) minh họa trực quan xác suất P(Real) và P(Fake).
* **Tùy chỉnh ngưỡng đánh giá (Threshold):** Cho phép thay đổi linh hoạt ngưỡng xác định Fake (mặc định là 0.5) thông qua tham số dòng lệnh.
* **Chế độ chạy ẩn (No-show):** Tích hợp cờ `--no-show` giúp chạy hàng loạt (batch inference) mà không bị gián đoạn bởi cửa sổ hiển thị đồ họ.

## 🧠 Model
Hệ thống sử dụng kiến trúc Transfer Learning với thông số cụ thể như sau:
* **Backbone:** `efficientnet_b3` được trích xuất từ thư viện `timm` (PyTorch Image Models).
* **Feature Dimension:** 1536 features đầu ra từ backbone.
* **Custom Classifier:** 
  * Cấu trúc Multi-Layer Perceptron (MLP) thu gọn.
  * Tích hợp BatchNormalization và ReLU activation sau mỗi lớp Linear.
  * Sử dụng Dropout (0.4 và 0.3) để giảm thiểu hiện tượng Overfitting.
  * Lớp đầu ra cuối cùng phân loại thành 2 classes (Real/Fake).

## 🔄 Pipeline
Quy trình tiền xử lý dữ liệu được thiết kế chặt chẽ thông qua thư viện `albumentations`:
1. **Đọc và chuyển đổi màu:** Ảnh được đọc và chuyển từ không gian màu BGR sang RGB bằng OpenCV.
2. **Resize:** Mọi ảnh đầu vào được đưa về kích thước chuẩn `224x224`.
3. **Normalize:** Chuẩn hóa dữ liệu theo phân phối chuẩn của ImageNet với `mean=[0.485, 0.456, 0.406]` và `std=[0.229, 0.224, 0.225]`.
4. **To Tensor:** Chuyển đổi định dạng ảnh thành PyTorch Tensor (`ToTensorV2`) để đẩy vào mô hình xử lý.

## 🛠️ Technologies
* **Ngôn ngữ:** Python
* **Deep Learning Framework:** PyTorch, Torchvision
* **Computer Vision / Image Processing:** OpenCV (`cv2`), Pillow (`PIL`)
* **Data Augmentation:** Albumentations
* **Pre-trained Models:** `timm`
* **Visualization:** Matplotlib

## 📂 Project Structure
```text
deepfake-detection/
├── weights/
│   └── best_model.pth        # https://drive.google.com/file/d/1n1BfdbCNzU5eaIzaHyQ9hqQXEp8qoVxh/view?usp=sharing
├── test_images/
│   └── real.jpg or fake.jpg            # Ảnh dùng để test
├── predict_deepfake.py       
├── requirements.txt          # Danh sách thư viện cần thiết
└── README.md
🚀 Installation
1. Clone kho lưu trữ này về máy tính

Bash
git clone [https://github.com/username-cua-ban/deepfake-detection.git](https://github.com/username-cua-ban/deepfake-detection.git)
cd deepfake-detection
2. Cài đặt các thư viện phụ thuộc

Bash
pip install -r requirements.txt
3. Cài đặt Model Weights
Do giới hạn dung lượng lưu trữ trên GitHub, vui lòng tải file best_model.pth tại link dưới đây và đặt vào thư mục weights/.

Link Google Drive tải best_model.pth # https://drive.google.com/file/d/1n1BfdbCNzU5eaIzaHyQ9hqQXEp8qoVxh/view?usp=sharing

▶️ Run Locally
Sau khi hoàn tất cài đặt, bạn có thể chạy dự đoán một bức ảnh bằng lệnh sau:

Bash
python predict_deepfake.py --image "test_images/fake.jpg" --model "weights/best_model.pth"
