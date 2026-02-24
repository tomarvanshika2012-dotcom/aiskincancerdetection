import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

st.set_page_config(page_title="AI Skin Cancer Detection", layout="wide")
st.title("🧠 AI Skin Cancer Detection System")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Select Mode
# -----------------------
mode = st.radio(
    "Select Detection Mode:",
    ("Binary Classification", "Multi-Class Classification")
)

if mode == "Binary Classification":
    class_names = ["Non-Cancer", "Cancer"]
else:
    class_names = [
        "Melanoma",
        "Basal Cell Carcinoma",
        "Squamous Cell Carcinoma",
        "Benign Nevus"
    ]

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model(num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("cancer_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(len(class_names))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# -----------------------
# Grad-CAM Function
# -----------------------
def generate_gradcam(model, image_tensor):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    final_layer = model.layer4
    final_layer.register_forward_hook(forward_hook)
    final_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)
    model.zero_grad()
    output[0, pred_class].backward()

    grad = gradients[0]
    act = activations[0]

    weights = torch.mean(grad, dim=[2,3], keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze()
    cam = F.relu(cam)

    cam = cam.cpu().detach().numpy()
    cam = cv2.resize(cam, (224,224))
    cam = cam - cam.min()
    cam = cam / cam.max()

    return cam

# -----------------------
# Upload Image
# -----------------------
uploaded_file = st.file_uploader("Upload Skin Lesion Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    prediction = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    st.subheader("🔍 Prediction Result")
    st.write(f"**Detected:** {prediction}")
    st.write(f"**Confidence:** {confidence_score:.2f}%")

    # Risk Level Indicator
    if confidence_score > 80:
        st.success("🟢 High Risk")
    elif confidence_score > 60:
        st.warning("🟡 Medium Risk")
    else:
        st.info("🔵 Low Risk")

    # Grad-CAM Heatmap
    st.subheader("🔥 Lesion Area Highlighting (Grad-CAM)")
    cam = generate_gradcam(model, img_tensor)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original = cv2.resize(np.array(image), (224,224))
    superimposed = heatmap * 0.4 + original
    st.image(superimposed.astype(np.uint8), caption="Grad-CAM Heatmap")

# -----------------------
# Model Performance Dashboard
# -----------------------
st.sidebar.title("📊 Model Performance Metrics")

# Example values (replace with real values from training)
accuracy = 0.91
precision = 0.89
recall = 0.94
f1 = 0.91

st.sidebar.write(f"Accuracy: {accuracy}")
st.sidebar.write(f"Precision: {precision}")
st.sidebar.write(f"Recall: {recall}")
st.sidebar.write(f"F1-Score: {f1}")

# Dummy Confusion Matrix (Replace with real one)
cm = np.array([[45,5],[3,47]])

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.sidebar.pyplot(fig)