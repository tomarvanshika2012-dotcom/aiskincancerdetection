import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Skin Cancer Detection",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 AI Skin Cancer Detection System")
st.write("Upload a skin lesion image to classify Cancer / Non-Cancer")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_trained_model():
    model_path = "skin_cancer_model.h5"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please train the model first.")
        st.stop()

    model = load_model(model_path)
    return model

model = load_trained_model()

# ===============================
# IMAGE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to model input size
    image = image.resize((224, 224))

    # Convert to array
    img_array = np.array(image)

    # Normalize (IMPORTANT)
    img_array = img_array.astype("float32") / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # ===============================
    # PREDICTION
    # ===============================
    prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction Result")
    st.write(f"Raw Probability: {prediction:.4f}")

    threshold = 0.7  # safer threshold to avoid false cancer detection

    if prediction >= threshold:
        st.error("⚠️ Cancer Detected")
        st.write(f"Confidence: {prediction * 100:.2f}%")

    elif prediction <= (1 - threshold):
        st.success("✅ Non-Cancer")
        st.write(f"Confidence: {(1 - prediction) * 100:.2f}%")

    else:
        st.warning("⚠️ Uncertain Prediction")
        st.write("Model is not confident. Please consult a doctor.")