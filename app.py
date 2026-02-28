import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 AI Skin Cancer Detection System")
st.write("Upload a skin lesion image to classify Cancer / Non-Cancer")

# =========================
# LOAD MODEL (FIXED SAFE VERSION)
# =========================
@st.cache_resource
def load_trained_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "skin_cancer_model.h5")

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please run train.py first.")
        st.stop()

    # IMPORTANT: compile=False prevents loading issues
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model

model = load_trained_model()

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image and force RGB
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize image
    img = img.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(img)

    # Safety checks
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Normalize
    img_array = img_array.astype("float32") / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(img_array)[0][0]

    st.subheader("🔍 Prediction Result")

    cancer_prob = float(prediction)
    non_cancer_prob = 1 - cancer_prob

    st.write(f"Raw Model Output: {cancer_prob:.4f}")

    if cancer_prob > 0.5:
        st.error("⚠️ Cancer Detected")
        st.write(f"Confidence: {cancer_prob * 100:.2f}%")
    else:
        st.success("✅ Non-Cancer")
        st.write(f"Confidence: {non_cancer_prob * 100:.2f}%")

    # Optional Probability Display
    st.progress(cancer_prob)
    st.caption("Cancer Probability Meter")