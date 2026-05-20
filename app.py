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
import os
import gdown

# -----------------------
# MODEL DOWNLOAD
# -----------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1Glb1f239ny-z9YChFObxYaG7yF2Vda6j"
MODEL_PATH = "best_effnetv2_rw_s.pth"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -----------------------
# CONFIG
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384

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
# GRAD-CAM (Correct for EfficientNetV2)
# -----------------------
last_conv_output = None
last_conv_grad = None

def save_activation(module, input, output):
    global last_conv_output
    last_conv_output = output

def save_gradient(module, grad_input, grad_output):
    global last_conv_grad
    last_conv_grad = grad_output[0]

# Hook the LAST CONV layer of EfficientNetV2-RW-S
target_layer = model.backbone.blocks[-1][-1].conv_pwl
target_layer.register_forward_hook(save_activation)
target_layer.register_full_backward_hook(save_gradient)

def generate_gradcam(img_tensor, gender_tensor):
    global last_conv_output, last_conv_grad

    last_conv_output = None
    last_conv_grad = None

    pred, logits = model(img_tensor, gender_tensor)
    pred.backward()

    activations = last_conv_output.detach().cpu().numpy()[0]
    gradients = last_conv_grad.detach().cpu().numpy()[0]

    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cam - cam.min()
    cam = cam / cam.max()

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(page_title="Bone Age Predictor", layout="wide")

st.markdown("""
    <style>
        .title {
            font-size: 42px;
            font-weight: 800;
            text-align: center;
            color: #0A3D62;
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #555;
            margin-bottom: 25px;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #E8F6F3;
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            color: #0E6251;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🩻 Bone Age Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a hand X-ray and view the model’s prediction and attention map.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])

gender_option = st.radio(
    "Select Gender (optional):",
    ("Not specified", "Male", "Female"),
    horizontal=True
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    img_np = np.array(image)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Predict Bone Age"):
        img_t = tfm(image=img_rgb)["image"]
        img_t = img_t.unsqueeze(0).to(DEVICE)

        if gender_option == "Male":
            gender_val = 1
        elif gender_option == "Female":
            gender_val = 0
        else:
            gender_val = 0

        gender_t = torch.tensor([gender_val], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            pred_months, logits = model(img_t, gender_t)

        st.markdown(
            f'<div class="prediction-box">Predicted Bone Age:<br>'
            f'<span style="font-size:32px;">{pred_months.item():.1f} months</span><br>'
            f'({pred_months.item()/12:.2f} years)</div>',
            unsafe_allow_html=True
        )

        heatmap = generate_gradcam(img_t, gender_t)
        overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)

        with col2:
            st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)
