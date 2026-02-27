import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from PIL import Image
import datetime
import os

# =========================
# REPORTLAB IMPORTS
# =========================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# =========================
# PAGE CONFIG (Responsive)
# =========================
st.set_page_config(
    page_title="AI Skin Cancer Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# THEME TOGGLE
# =========================
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        body {background-color: #0E1117; color: white;}
        </style>
    """, unsafe_allow_html=True)

# =========================
# SIDEBAR NAVIGATION
# =========================
menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Detection", "About"]
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_my_model():
    return load_model("skin_cancer_model.h5")

model = load_my_model()

# =========================
# HOME PAGE
# =========================
if menu == "Home":

    col1, col2 = st.columns([1, 4])

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966483.png", width=100)

    with col2:
        st.title("AI Skin Cancer Detection System")
        st.write("Early detection using Deep Learning.")

    st.markdown("---")
    st.info("Upload a skin image to analyze cancer risk.")

# =========================
# DETECTION PAGE
# =========================
elif menu == "Detection":

    st.header("Upload Skin Image")

    threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5)

    # Upload options
    upload_option = st.radio("Select Input Method:", ["Upload Image", "Use Camera"])

    uploaded_file = None

    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader(
            "Drag and Drop or Browse Image",
            type=["jpg", "jpeg", "png"]
        )

    elif upload_option == "Use Camera":
        uploaded_file = st.camera_input("Capture Skin Image")

    if uploaded_file is not None:

        # Convert to RGB (fix 4 channel issue)
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Preview Image", use_column_width=True)

        # Remove Image Button
        if st.button("Remove Image"):
            st.experimental_rerun()

        # =========================
        # PREPROCESS
        # =========================
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # =========================
        # PREDICTION
        # =========================
        prediction = model.predict(img_array)
        prob = float(prediction[0][0])

        st.subheader("Prediction Result")
        st.write("Raw Prediction Value:", prob)

        if prob > threshold:
            result = "Cancer Detected"
            st.error(result)
        else:
            result = "Non-Cancer"
            st.success(result)

        confidence = prob if prob > 0.5 else 1 - prob
        st.write(f"Confidence: {confidence*100:.2f}%")
        st.progress(int(confidence * 100))

        st.subheader("Probability Distribution")
        st.write(f"Cancer Probability: {prob*100:.2f}%")
        st.write(f"Non-Cancer Probability: {(1-prob)*100:.2f}%")

        if prob < 0.4:
            st.success("Low Risk")
        elif prob < 0.7:
            st.warning("Medium Risk")
        else:
            st.error("High Risk")

        # =========================
        # PDF REPORT
        # =========================
        if st.button("Generate PDF Report"):

            file_name = "Skin_Cancer_Report.pdf"
            doc = SimpleDocTemplate(file_name, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("<b>AI Skin Cancer Detection Report</b>", styles["Title"]))
            elements.append(Spacer(1, 0.5 * inch))

            current_time = datetime.datetime.now()

            elements.append(Paragraph(f"Date: {current_time}", styles["Normal"]))
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph(f"Prediction: {result}", styles["Normal"]))
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles["Normal"]))
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("Disclaimer: Educational use only.", styles["Normal"]))

            doc.build(elements)

            with open(file_name, "rb") as pdf_file:
                st.download_button(
                    label="Download PDF",
                    data=pdf_file,
                    file_name=file_name,
                    mime="application/pdf"
                )

# =========================
# ABOUT PAGE
# =========================
elif menu == "About":

    st.title("About This Project")
    st.write("""
    This AI system detects potential skin cancer using a trained deep learning model.
    
    ⚠ Disclaimer: This tool is for educational purposes only.
    It is not a replacement for medical diagnosis.
    """)