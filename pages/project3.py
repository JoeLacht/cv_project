import streamlit as st
import torch
import torch.nn as nn
import packaging.version
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        bridge = self.bridge(p2)
        u2 = self.up2(bridge)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))
        return torch.sigmoid(self.final(c1))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# --- Загрузка весов ---
torch_version = packaging.version.parse(torch.__version__)
if torch_version >= packaging.version.parse("2.6"):
    state_dict = torch.load("models/best_unet2.pth", map_location=device, weights_only=False)
else:
    state_dict = torch.load("models/best_unet2.pth", map_location=device)

model.load_state_dict(state_dict)
model.eval()

# ----------- Преобразования -----------
transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(),
    ToTensorV2()
])

# ----------- Интерфейс Streamlit -----------
st.title("🌲 Forest Segmentation with U-Net")
st.write("Загрузите изображения или вставьте ссылку для сегментации леса.")

uploaded_files = st.file_uploader("Выберите изображения", type=["jpg","png","jpeg"], accept_multiple_files=True)
url_input = st.text_input("Или вставьте ссылку на изображение")

def load_image_from_url(url):
    import requests
    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return np.array(img)

images_to_predict = []
if uploaded_files:
    for file in uploaded_files:
        img = np.array(Image.open(file).convert("RGB"))
        images_to_predict.append(img)
if url_input:
    try:
        img = load_image_from_url(url_input)
        images_to_predict.append(img)
    except:
        st.warning("Не удалось загрузить изображение по ссылке.")

if st.button("Определить лес"):
    if not images_to_predict:
        st.warning("Сначала загрузите хотя бы одно изображение.")
    for i, img_orig in enumerate(images_to_predict):
        augmented = transform(image=img_orig)
        img_tensor = augmented["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred_mask = model(img_tensor)[0, 0].cpu().numpy()
            pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

        # Денормализация для отображения
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_vis = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        img_vis = (img_vis * std + mean)
        img_vis = np.clip(img_vis, 0, 1)

        # Overlay зелёной маски
        overlay = img_vis.copy()
        alpha = 0.5
        overlay[pred_mask_bin == 1] = (
            (1 - alpha) * overlay[pred_mask_bin == 1] + alpha * np.array([0, 1, 0])
        )
        # Средняя уверенность
        confidence = pred_mask.mean()
        st.write(f"🖼 Изображение {i+1}, средняя уверенность модели: {confidence:.2f}")

        # Отображение результата
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_vis)
        axes[0].axis("off")
        axes[0].set_title("Original Image")
        axes[1].imshow(overlay)
        axes[1].axis("off")
        axes[1].set_title("Predicted Mask Overlay")
        st.pyplot(fig)