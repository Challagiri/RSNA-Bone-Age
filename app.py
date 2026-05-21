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
# STRICT HAND X-RAY VALIDATION
# -----------------------
def is_hand_xray_strict(pil_img):
    img = np.array(pil_img)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    brightness = np.mean(gray)
    contrast = np.std(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)

    # Tuned thresholds – adjust if needed
    if brightness < 170 and contrast > 30 and edge_density > 0.02:
        return True
    return False


# -----------------------
# REGION DETECTION (CARPAL / METACARPALS / PHALANGES)
# -----------------------
def detect_focus_region(heatmap):
    h = heatmap.shape[0]
    zone_h = h // 3

    phalanges_zone = heatmap[0:zone_h, :, :]
    metacarpals_zone = heatmap[zone_h:2*zone_h, :, :]
    carpals_zone = heatmap[2*zone_h:h, :, :]

    ph = np.sum(phalanges_zone)
    mt = np.sum(metacarpals_zone)
    cp = np.sum(carpals_zone)

    regions = {
        "Phalanges (Fingers)": ph,
        "Metacarpals (Palm)": mt,
        "Carpal Bones (Wrist)": cp
    }

    focus_region = max(regions, key=regions.get)
    return focus_region, regions


# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Bone Age • Cinematic Grad-CAM", layout="wide")

# -----------------------
# GLOBAL CINEMATIC STYLE
# -----------------------
st.markdown("""
<style>

body {
    background: radial-gradient(circle at 20% 20%, #050816 0%, #0b1220 40%, #020617 100%);
    background-size: 200% 200%;
    animation: gradientShift 12s ease infinite;
    color: #f5f5f5;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.fade-in {
    animation: fadeIn 1.0s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Centered card */
.center-card {
    margin: 0 auto;
    max-width: 520px;
    background: rgba(15,23,42,0.85);
    border-radius: 18px;
    padding: 22px;
    border: 1px solid rgba(56,189,248,0.6);
    box-shadow: 0 0 35px rgba(0,255,255,0.35);
    backdrop-filter: blur(14px);
}

/* Info card */
.info-card {
    margin: 18px auto 26px auto;
    max-width: 820px;
    background: rgba(0,255,255,0.08);
    border-radius: 16px;
    padding: 18px 22px;
    border: 1px solid rgba(0,255,255,0.45);
    box-shadow: 0 0 28px rgba(0,255,255,0.35);
    backdrop-filter: blur(10px);
}

/* Buttons */
.stButton>button {
    border-radius: 999px;
    border: 1px solid rgba(56,189,248,0.7);
    background: radial-gradient(circle at top left, #0ea5e9 0, #0369a1 40%, #020617 100%);
    color: #E5F6FF;
    font-weight: 600;
    padding: 0.55rem 1rem;
    box-shadow: 0 0 22px rgba(56,189,248,0.65);
    transition: all 0.25s ease-in-out;
}

.stButton>button:hover {
    transform: scale(1.06);
    box-shadow: 0 0 40px rgba(0,255,255,0.9);
}

/* Prediction box */
.prediction-box {
    padding: 18px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(0, 191, 255, 0.16), rgba(0, 255, 200, 0.10));
    border: 1px solid rgba(0, 191, 255, 0.55);
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    color: #E8F8F5;
    margin-top: 18px;
    box-shadow: 0 0 28px rgba(0, 191, 255, 0.35);
}

.prediction-box span {
    font-size: 32px;
    color: #00E5FF;
}

/* Grad-CAM titles */
.gradcam-title {
    font-size: 17px;
    font-weight: 600;
    color: #E5E7EB;
    margin-bottom: 6px;
}

/* Popup for invalid image */
.popup-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0,0,0,0.75);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}
.result-box {
    transition: transform 0.3s ease-in-out;
}
.result-box:hover {
    transform: scale(1.03);
    box-shadow: 0 0 40px rgba(0,255,255,0.8);
}

.popup-box {
    background: rgba(15,23,42,0.96);
    padding: 26px;
    border-radius: 18px;
    border: 1px solid rgba(56,189,248,0.8);
    box-shadow: 0 0 35px rgba(0,255,255,0.6);
    backdrop-filter: blur(14px);
    color: #E0FFFF;
    text-align: center;
    width: 380px;
    animation: fadeIn 0.8s ease-out;
}

/* Footer */
.footer-text {
    text-align: center;
    font-size: 13px;
    color: #9CA3AF;
    margin-top: 26px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# HEADER
# -----------------------
st.markdown(
    '<div class="fade-in" style="font-size:46px;font-weight:900;text-align:center;color:#EAF2F8;letter-spacing:0.06em;text-shadow:0 0 25px rgba(0,255,255,0.9);">BONE AGE • GRAD‑CAM LAB</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="fade-in" style="text-align:center;color:#D0D3D4;margin-bottom:18px;">Upload a hand X‑ray, estimate bone age in months, and see where the model is focusing.</div>',
    unsafe_allow_html=True,
)

# -----------------------
# MODEL & CONCEPT INFO
# -----------------------
st.markdown("""
<div class="info-card fade-in">
    <h3 style="color:#00E5FF; margin-top:0;">🧬 About this model</h3>
    <p style="color:#E0FFFF; margin-bottom:6px;">
        This AI system uses <b>EfficientNetV2‑RW‑S</b>, a deep convolutional neural network,
        trained on thousands of pediatric hand X‑rays to estimate <b>bone age</b>.
    </p>
    <p style="color:#E0FFFF; margin-bottom:6px;">
        It looks at bone growth plates, joint spacing, and shape of the <b>carpal bones</b>,
        <b>metacarpals</b>, and <b>phalanges</b>. The <b>Grad‑CAM</b> heatmap shows which region
        the model is focusing on when making its prediction.
    </p>
    <p style="color:#B2EBF2; margin-bottom:0;">
        This demo is for <b>educational purposes only</b> and is not intended for clinical diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------
# SMALL CENTERED UPLOAD BOX WITH UPLOADER INSIDE
# -----------------------
st.markdown("""
<div style="
    margin: 0 auto;
    max-width: 220px;
    background: rgba(15,23,42,0.85);
    border-radius: 14px;
    padding: 14px 16px;
    border: 1px solid rgba(56,189,248,0.6);
    box-shadow: 0 0 20px rgba(0,255,255,0.25);
    backdrop-filter: blur(10px);
    text-align:center;
">
    <h3 style="color:#E5F6FF; font-size:18px; margin:0 0 8px 0;">🦴 Upload Hand X‑ray</h3>
</div>
""", unsafe_allow_html=True)

# Center the uploader directly below the heading
st.markdown("<div style='display:flex; justify-content:center; margin-top:8px;'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a PA view hand X‑ray (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
    key="file_upload"
)

st.markdown("</div>", unsafe_allow_html=True)


gender_option = st.radio(
    "Patient gender (optional):",
    ("Not specified", "Male", "Female"),
    horizontal=True,
)

run_button = st.button("🔍 Run Bone Age Prediction", use_container_width=250)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# MAIN LOGIC
# -----------------------
if run_button:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a hand X‑ray image first.")
        st.stop()

    orig_image = Image.open(uploaded_file).convert("RGB")

    if not is_hand_xray_strict(orig_image):
        st.error("⚠️ Invalid Image — This does not appear to be a hand X‑ray.")
        st.stop()

    st.markdown("### 📷 Uploaded Hand X‑ray (Preview)")
    st.image(orig_image, width=250)


    image = orig_image.convert("L")
    img_np = np.array(image)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    img_t = tfm(image=img_rgb)["image"]
    img_t = img_t.unsqueeze(0).to(DEVICE)

    gender_val = 1 if gender_option == "Male" else 0
    gender_t = torch.tensor([gender_val], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        pred_months, logits = model(img_t, gender_t)

    months_val = pred_months.item()
    years_val = months_val / 12.0


    heatmap = generate_gradcam(img_t, gender_t)

    img_resized = cv2.resize(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        (IMG_SIZE, IMG_SIZE)
    )
    overlay = cv2.addWeighted(img_resized, 0.5, heatmap, 0.5, 0)

    focus_region, region_scores = detect_focus_region(heatmap)

    # -----------------------
    # RESULT BOXES (SIDE-BY-SIDE SMALL BOXES)
    # -----------------------
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown(f"""
        <div style="
            padding: 16px;
            border-radius: 14px;
            background: rgba(0, 255, 255, 0.08);
            border: 1px solid rgba(0, 255, 255, 0.45);
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.55);
            text-align: center;
            backdrop-filter: blur(10px);
            animation: fadeIn 1.0s ease-in-out;
            margin-top: 10px;
        ">
            <div style="font-size: 24px; font-weight: 700; color: #E0FFFF;">
                📊 Predicted Bone Age
            </div>
            <div style="font-size: 26px; font-weight: 800; color: #00E5FF; margin-top: 6px;">
                {months_val:.1f} months
            </div>
            <div style="font-size: 18px; color: #B2EBF2;">
                ({years_val:.2f} years)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with colB:
        st.markdown(f"""
        <div style="
            padding: 16px;
            border-radius: 14px;
            background: rgba(0, 255, 255, 0.08);
            border: 1px solid rgba(0, 255, 255, 0.45);
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.55);
            text-align: center;
            backdrop-filter: blur(10px);
            animation: fadeIn 1.0s ease-in-out;
            margin-top: 10px;
        ">
            <div style="font-size: 24px; font-weight: 700; color: #E0FFFF;">
                🧠 Model Focus Area
            </div>
            <div style="font-size: 18px; color: #B2EBF2; margin-top: 6px;">
                The model is primarily focusing on:
            </div>
            <div style="font-size: 24px; font-weight: 800; color: #00E5FF; margin-top: 6px;">
                {focus_region}
            </div>
        </div>
        """, unsafe_allow_html=True)


    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div style="text-align:center;" class="gradcam-title fade-in">Original (Processed)</div>',unsafe_allow_html=True,)
        st.image(img_resized[:, :, ::-1], width=560)

    with col2:
        st.markdown('<div style="text-align:center;" class="gradcam-title fade-in">Grad‑CAM Heatmap</div>',unsafe_allow_html=True,)
        st.image(overlay[:, :, ::-1], width=560)

# -----------------------
# FOOTER
# -----------------------
st.markdown(
    '<div class="footer-text">Bone Age demo • Grad‑CAM visualization for educational purposes only.</div>',
    unsafe_allow_html=True,
)
