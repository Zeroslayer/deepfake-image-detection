
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

try:
    import timm
except ImportError:
    print("Cai dat thu vien: pip install timm")
    sys.exit(1)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Cai dat thu vien: pip install albumentations")
    sys.exit(1)


class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b3', pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features  # 1536

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


# ======================================================
#  LOAD MODEL
# ======================================================
def load_model(model_path, device):
    if not os.path.exists(model_path):
        print(f" Khong tim thay file model tai: {model_path}")
        print(" Huong dan: Vui long tai file 'best_model.pth' tu link Google Drive trong README.md")
        print("   sau do dat vao cung thu muc hoac thu muc 'weights/'.")
        sys.exit(1)

    print("Loading model: " + model_path)
    checkpoint = torch.load(model_path, map_location=device)

    model = DeepfakeDetector(num_classes=2, pretrained=False, freeze_backbone=False)

    # Notebook save theo format: {'model_state': ..., 'epoch': ..., 'val_acc': ...}
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        epoch   = checkpoint.get('epoch', '?')
        val_acc = checkpoint.get('val_acc', 0)
        print("Load OK! Epoch=" + str(epoch) + " | Val_Acc=" + str(round(val_acc * 100, 2)) + "%")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Load OK! (key=state_dict)")
    else:
        # Truong hop save thang state_dict
        model.load_state_dict(checkpoint)
        print("Load OK!")

    model.to(device)
    model.eval()
    return model


# ======================================================
#  PREDICT
# label_map trong notebook: real=0, fake=1
# ======================================================
def predict_image(image_path, model, device, threshold=0.5):
    if not os.path.exists(image_path):
        print("Khong tim thay anh: " + image_path)
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print("Khong doc duoc anh (sai dinh dang?): " + image_path)
        sys.exit(1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = val_transform(image=img)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)                        # raw scores
        probs  = torch.softmax(logits, dim=1)[0]     # xac suat

    # probs[0] = Real, probs[1] = Fake  (theo label_map cua notebook)
    real_prob = probs[0].item()
    fake_prob = probs[1].item()

    print("\n[DEBUG] logits : " + str([round(logits[0][0].item(), 4), round(logits[0][1].item(), 4)]))
    print("[DEBUG] P(Real): " + str(round(real_prob * 100, 2)) + "%")
    print("[DEBUG] P(Fake): " + str(round(fake_prob * 100, 2)) + "%")
    print("[DEBUG] Threshold: " + str(threshold))

    if fake_prob >= threshold:
        label      = 'Fake'
        confidence = fake_prob * 100
    else:
        label      = 'Real'
        confidence = real_prob * 100

    return label, confidence, fake_prob, real_prob


# ======================================================
# 5. HIEN THI KET QUA
# ======================================================
def show_result(image_path, label, confidence, fake_prob, real_prob):
    img   = Image.open(image_path).convert('RGB')
    color = '#EA580C' if label == 'Fake' else '#2563EB'
    title = ('FAKE' if label == 'Fake' else 'REAL') + '  |  ' + str(round(confidence, 1)) + '%'

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].imshow(img)
    axes[0].set_title(title, color=color, fontweight='bold', fontsize=14)
    axes[0].axis('off')

    bars = axes[1].barh(['Real', 'Fake'],
                        [real_prob * 100, fake_prob * 100],
                        color=['#2563EB', '#EA580C'])
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel('Xac suat (%)')
    axes[1].set_title('Phan bo xac suat', fontsize=11, fontweight='bold')
    for i, v in enumerate([real_prob * 100, fake_prob * 100]):
        axes[1].text(v + 1, i, str(round(v, 1)) + '%', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


# ======================================================
# 6. MAIN
# ======================================================
def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Duong dan anh can kiem tra')
    parser.add_argument('--model', type=str,
                        default='weights/best_model.pth',
                        help='Duong dan file best_model.pth (mac dinh: weights/best_model.pth)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Nguong Fake (mac dinh 0.5)')
    parser.add_argument('--no-show', action='store_true',
                        help='Khong hien thi anh')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    model = load_model(args.model, device)

    print("\nDang phan tich: " + args.image)
    label, confidence, fake_prob, real_prob = predict_image(
        args.image, model, device, threshold=args.threshold)

    print("")
    print("=" * 42)
    ket_qua = "GIA (FAKE)" if label == "Fake" else "THAT (REAL)"
    print("  Ket qua    : " + ket_qua)
    print("  Do tin cay : " + str(round(confidence, 2)) + "%")
    print("  P(Real)    : " + str(round(real_prob * 100, 2)) + "%")
    print("  P(Fake)    : " + str(round(fake_prob * 100, 2)) + "%")
    print("  Threshold  : " + str(args.threshold))
    print("=" * 42)

    if not args.no_show:
        show_result(args.image, label, confidence, fake_prob, real_prob)


if __name__ == '__main__':
    main()