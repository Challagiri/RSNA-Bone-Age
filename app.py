import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
# GRAD-CAM HOOKS
# -----------------------
last_conv_output = None
last_conv_grad = None


def save_activation(module, input, output):
    global last_conv_output
    last_conv_output = output


def save_gradient(module, grad_input, grad_output):
    global last_conv_grad
    last_conv_grad = grad_output[0]


# Hook last conv layer of EfficientNetV2-RW-S
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
    cam = cam / (cam.max() + 1e-8)

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap


# -----------------------
# SIMPLE X-RAY VALIDATION
# -----------------------
def is_valid_xray(img_pil):
    gray = np.array(img_pil)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    # Heuristic: X-rays are darker with strong contrast (bones vs background)
    if brightness < 170 and contrast > 25:
        return True
    return False


# -----------------------
# STREAMLIT UI CONFIG
# -----------------------
st.set_page_config(page_title="Bone Age AI • Grad-CAM", layout="wide")

# Global dark + gradient cinematic style
st.markdown(
    """
    <style>
    body {
        background: radial-gradient(circle at top, #1b2735 0, #090a0f 55%, #000000 100%);
        color: #f5f5f5;
    }
    .main {
        background: transparent;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        color: #EAF2F8;
        margin-bottom: 4px;
        text-shadow: 0 0 18px rgba(0, 191, 255, 0.7);
    }
    .subtitle {
        font-size: 17px;
        text-align: center;
        color: #D0D3D4;
        margin-bottom: 25px;
    }
    .prediction-box {
        padding: 18px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(0, 191, 255, 0.12), rgba(0, 255, 200, 0.08));
        border: 1px solid rgba(0, 191, 255, 0.4);
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        color: #E8F8F5;
        margin-top: 18px;
        box-shadow: 0 0 25px rgba(0, 191, 255, 0.25);
    }
    .prediction-box span {
        font-size: 30px;
        color: #00E5FF;
    }
    .side-card {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        border: 1px solid rgba(148, 163, 184, 0.4);
        box-shadow: 0 0 25px rgba(15, 23, 42, 0.9);
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #E5E7EB;
        margin-bottom: 8px;
    }
    .footer-text {
        text-align: center;
        font-size: 13px;
        color: #9CA3AF;
        margin-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# HEADER
# -----------------------
st.markdown('<div class="title">🩻 Bone Age AI • Grad‑CAM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a hand X‑ray, estimate bone age in months, and see where the model is focusing.</div>',
    unsafe_allow_html=True,
)

# -----------------------
# LAYOUT
# -----------------------
left, right = st.columns([1.1, 1])

with left:
    st.markdown('<div class="side-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">1️⃣ Upload Hand X‑ray</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Supported formats: PNG, JPG, JPEG",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    gender_option = st.radio(
        "Patient gender (optional):",
        ("Not specified", "Male", "Female"),
        horizontal=True,
    )

    predict_button = st.button("🔍 Run Bone Age Prediction", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="side-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">2️⃣ Model Output</div>', unsafe_allow_html=True)
    output_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# MAIN LOGIC
# -----------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    img_np = np.array(image)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    # Show original X-ray
    with left:
        st.image(image, caption="Uploaded Hand X‑ray", use_column_width=True)

    if predict_button:
        # Validate X-ray
        if not is_valid_xray(image):
            with right:
                output_placeholder.error(
                    "⚠️ This does not look like a typical hand X‑ray.\n\n"
                    "Please upload a clear hand X‑ray image to estimate bone age."
                )
        else:
            # Preprocess
            img_t = tfm(image=img_rgb)["image"]
            img_t = img_t.unsqueeze(0).to(DEVICE)

            if gender_option == "Male":
                gender_val = 1
            elif gender_option == "Female":
                gender_val = 0
            else:
                gender_val = 0

            gender_t = torch.tensor([gender_val], dtype=torch.long, device=DEVICE)

            # Prediction
            with torch.no_grad():
                pred_months, logits = model(img_t, gender_t)

            # Display prediction
            with right:
                months_val = pred_months.item()
                years_val = months_val / 12.0
                output_placeholder.markdown(
                    f"""
                    <div class="prediction-box">
                        Predicted Bone Age<br>
                        <span>{months_val:.1f} months</span><br>
                        ({years_val:.2f} years)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Grad-CAM
            heatmap = generate_gradcam(img_t, gender_t)

            # Resize original to match heatmap size to avoid cv2 error
            img_resized = cv2.resize(
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                (IMG_SIZE, IMG_SIZE)
            )
            overlay = cv2.addWeighted(img_resized, 0.5, heatmap, 0.5, 0)

            # Side-by-side visualization
            vis_col1, vis_col2 = st.columns(2)
            with vis_col1:
                st.markdown("#### Original X‑ray")
                st.image(img_resized[:, :, ::-1], use_column_width=True)
            with vis_col2:
                st.markdown("#### Grad‑CAM Focus Map")
                st.image(overlay[:, :, ::-1], use_column_width=True)

else:
    with right:
        output_placeholder.info(
            "Upload a hand X‑ray on the left and click **Run Bone Age Prediction** "
            "to see the model’s estimate and attention map."
        )

# -----------------------
# FOOTER
# -----------------------
st.markdown(
    '<div class="footer-text">Bone Age AI demo • Grad‑CAM visualization for educational purposes only, not for clinical use.</div>',
    unsafe_allow_html=True,
)
