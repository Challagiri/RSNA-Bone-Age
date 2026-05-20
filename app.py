import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
import os
import gdown

MODEL_URL = "https://drive.google.com/uc?export=download&id=1Glb1f239ny-z9YChFObxYaG7yF2Vda6j"
MODEL_PATH = "best_effnetv2_rw_s.pth"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384
MODEL_PATH = "best_effnetv2_rw_s.pth"

# Preprocessing
tfm = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.ToFloat(max_value=255.0),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# -----------------------
# MODEL DEFINITION
# -----------------------
class EffNetV2RWOrdinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnetv2_rw_s",
                                          pretrained=False,
                                          num_classes=0)
        self.gender_embed = nn.Embedding(2, 16)
        self.fc = nn.Sequential(
            nn.Linear(self.backbone.num_features + 16, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 240)
        )

    def forward(self, x, gender):
        f = self.backbone(x)
        g = self.gender_embed(gender)
        logits = self.fc(torch.cat([f, g], dim=1))
        prob = torch.softmax(logits, dim=1)
        months = torch.arange(240, device=logits.device, dtype=torch.float32)
        return (prob * months).sum(dim=1), logits


@st.cache_resource
def load_model():
    model = EffNetV2RWOrdinal().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# -----------------------
# GRAD-CAM FUNCTION
# -----------------------
def generate_gradcam(model, img_tensor, gender_tensor):
    img_tensor.requires_grad = True

    pred, logits = model(img_tensor, gender_tensor)
    pred.backward()

    # Extract gradients & activations
    gradients = model.backbone.get_classifier().weight.grad
    activations = model.backbone.forward_features(img_tensor)

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = activations.detach().cpu().numpy()[0]

    for i in range(len(pooled_gradients)):
        activations[i, :, :] *= pooled_gradients[i].cpu().numpy()

    heatmap = np.mean(activations, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(page_title="Bone Age Predictor", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: 700;
            text-align: center;
            color: #1F618D;
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 18px;
            text-align: center;
            color: #555;
            margin-bottom: 25px;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #EBF5FB;
            text-align: center;
            font-size: 22px;
            font-weight: 600;
            color: #154360;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🩻 Bone Age Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a hand X-ray and let the model estimate bone age.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])

gender_option = st.radio(
    "Select Gender (optional):",
    ("Not specified", "Male", "Female"),
    horizontal=True
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Predict Bone Age"):
        img_np = np.array(image)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        img_t = tfm(image=img_rgb)["image"]
        img_t = img_t.unsqueeze(0).to(DEVICE)

        # Gender handling
        if gender_option == "Male":
            gender_val = 1
        elif gender_option == "Female":
            gender_val = 0
        else:
            gender_val = 0  # default

        gender_t = torch.tensor([gender_val], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            pred_months, logits = model(img_t, gender_t)

        st.markdown(
            f'<div class="prediction-box">Predicted Bone Age:<br>'
            f'<span style="font-size:30px;">{pred_months.item():.1f} months</span><br>'
            f'({pred_months.item()/12:.2f} years)</div>',
            unsafe_allow_html=True
        )

        # -----------------------
        # SHOW GRAD-CAM
        # -----------------------
        st.subheader("Model Attention (Grad-CAM)")
        heatmap = generate_gradcam(model, img_t, gender_t)

        overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)

        st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)
