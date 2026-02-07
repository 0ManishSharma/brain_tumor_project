import streamlit as st
import os
from src.Brain_tumor.pipelines.prediction_pipeline import Prediction

from PIL import Image

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection using CNN")
st.write("Upload an MRI image to predict the tumor type.")

# ---------------------------
# Paths
# ---------------------------
MODEL_PATH = "artifacts/model.h5"
classes = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

# ---------------------------
# Load Predictor
# ---------------------------
@st.cache_resource
def load_predictor():
    return Prediction(MODEL_PATH,classes)

predictor = load_predictor()

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing MRI image..."):
            # save temp image
            temp_path = "temp_image.png"
            image.save(temp_path)

            prediction, confidence = predictor.predict(temp_path)

            st.success(f"🧠 Prediction: **{prediction.upper()}**")
            st.info(f"Confidence: **{confidence * 100:.2f}%**")

            os.remove(temp_path)
