import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

st.set_page_config(
    page_title="AI Skin Cancer Detection",
    page_icon="🧬",
    layout="wide"
)

# ================= SESSION INIT =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= SIDEBAR =================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
st.sidebar.title("🧬 AI Skin Cancer System")

page = st.sidebar.radio("🧭 Navigation", ["🏠 Home", "🔬 Prediction"])
theme = st.sidebar.radio("🎨 Theme Mode", ["Light", "Dark"])

# 1️⃣ Detection Threshold Slider
threshold = st.sidebar.slider("🎯 Detection Threshold", 0.0, 1.0, 0.5, 0.05)

# 2️⃣ Adjustable Sensitivity Mode
sensitivity_mode = st.sidebar.selectbox(
    "🧠 Sensitivity Mode",
    ["Normal", "High Sensitivity", "Low Sensitivity"]
)

# 4️⃣ Show/Hide Advanced Toggle
show_advanced = st.sidebar.checkbox("📊 Show Advanced Details")

# 3️⃣ Reset Button
if st.sidebar.button("🔄 Reset Prediction"):
    st.session_state.clear()
    st.rerun()

# ================= DARK MODE =================
if theme == "Dark":
    st.markdown("""
        <style>
        .stApp {background-color: #0E1117; color: white;}
        </style>
    """, unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model_path = "skin_cancer_model.h5"
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

# ================= PREPROCESS =================
def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# ================= GRAD-CAM =================
def generate_gradcam(image):
    img_array = preprocess(image)

    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# ================= PREDICT =================
def predict_image(image):
    arr = preprocess(image)
    prediction = model.predict(arr)[0][0]
    cancer_prob = float(prediction)
    non_cancer_prob = 1 - cancer_prob

    # Sensitivity Adjustment
    if sensitivity_mode == "High Sensitivity":
        adj_threshold = threshold - 0.1
    elif sensitivity_mode == "Low Sensitivity":
        adj_threshold = threshold + 0.1
    else:
        adj_threshold = threshold

    predicted = "Cancer" if cancer_prob > adj_threshold else "Non-Cancer"
    confidence = cancer_prob*100 if predicted=="Cancer" else non_cancer_prob*100

    return predicted, cancer_prob, non_cancer_prob, confidence, adj_threshold

# ================= PDF REPORT =================
def generate_pdf(patient_data, predicted, cancer_prob, confidence,
                 original_img, heatmap_img, overlay_img):

    filename = "AI_Skin_Cancer_Report.pdf"
    doc = SimpleDocTemplate(filename)
    elements = []
    styles = getSampleStyleSheet()

    now = datetime.now().strftime("%d-%m-%Y %H:%M")
    risk = cancer_prob * 100

    if risk <= 40:
        risk_level = "LOW RISK"
    elif risk <= 70:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "HIGH RISK"

    elements.append(Paragraph("AI Skin Cancer Detection Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    data = [
        ["Patient Name", patient_data["name"]],
        ["Age", patient_data["age"]],
        ["Gender", patient_data["gender"]],
        ["Skin Type", patient_data["skin_type"]],
        ["Prediction", predicted],
        ["Confidence", f"{confidence:.2f}%"],
        ["Risk Level", risk_level],
        ["Date & Time", now]
    ]

    table = Table(data, colWidths=[220, 200])
    table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.grey)]))
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    # Save images
    cv2.imwrite("original.png", cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite("heatmap.png", heatmap_img)
    cv2.imwrite("overlay.png", overlay_img)

    elements.append(Paragraph("Original Image", styles["Heading2"]))
    elements.append(RLImage("original.png", width=3*inch, height=3*inch))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Grad-CAM Heatmap", styles["Heading2"]))
    elements.append(RLImage("heatmap.png", width=3*inch, height=3*inch))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Overlay Visualization", styles["Heading2"]))
    elements.append(RLImage("overlay.png", width=3*inch, height=3*inch))
    elements.append(Spacer(1, 0.3 * inch))

    # Probability graph
    plt.figure()
    plt.bar(["Cancer", "Non-Cancer"],
            [cancer_prob*100, (1-cancer_prob)*100])
    plt.ylim(0,100)
    plt.ylabel("Probability (%)")
    plt.tight_layout()
    plt.savefig("prob.png")
    plt.close()

    elements.append(Paragraph("Probability Graph", styles["Heading2"]))
    elements.append(RLImage("prob.png", width=4*inch, height=2.5*inch))

    doc.build(elements)
    return filename

# ================= HOME =================
if page == "🏠 Home":
    st.title("🧬 AI Skin Cancer Detection System")
    st.warning("⚠ Educational use only. Not a medical diagnosis tool.")

# ================= PREDICTION =================
if page == "🔬 Prediction":

    st.title("🔬 Skin Cancer Prediction")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Patient Name")
        age = st.number_input("Age", 1, 120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    with col2:
        skin_type = st.selectbox("Skin Type", ["Fair", "Medium", "Dark"])

    img1 = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    if img1:
        image = Image.open(img1).convert("RGB")
        st.image(image, use_column_width=True)

        if st.button("🔍 Predict"):

            with st.spinner("Analyzing Image..."):
                predicted, cancer_prob, non_cancer_prob, confidence, adj_threshold = predict_image(image)

            st.success(f"Prediction: {predicted}")
            st.write(f"Confidence: {confidence:.2f}%")
            st.progress(int(confidence))

            # Grad-CAM
            heatmap = generate_gradcam(image)
            heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(np.array(image), 0.6, heatmap_color, 0.4, 0)

            st.image(overlay, caption="Suspicious Area Highlight")

            if show_advanced:
                st.write(f"Non-Cancer Probability: {non_cancer_prob*100:.2f}%")
                st.write(f"Threshold Used: {adj_threshold}")
                st.write(f"Sensitivity Mode: {sensitivity_mode}")

            patient_data = {
                "name": name,
                "age": age,
                "gender": gender,
                "skin_type": skin_type
            }

            pdf = generate_pdf(patient_data, predicted, cancer_prob,
                               confidence, np.array(image),
                               heatmap_color, overlay)

            with open(pdf, "rb") as f:
                st.download_button("📄 Download Full Report", f, file_name=pdf)

st.markdown("---")
st.markdown("© 2026 AI Medical Assistant | Gautam Buddha University")