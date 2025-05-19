import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from datetime import datetime
import subprocess

st.set_page_config(
    page_title="Digit Classifier with Feedback & Retraining",
    layout="centered",
    page_icon="🧠"
)

st.title("🔢 Handwritten Digit Classifier with Feedback & Retraining")
st.markdown(
    "Upload a **28x28 grayscale image** of a digit (0–9). The model will predict the digit.\n"
    "If the prediction is wrong, provide the correct label and save it for retraining."
)

MODEL_PATH = 'digitmodel.h5'
TRAIN_LABEL_DIR = "train_label"

@st.cache_resource
def load_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    img = image.convert('L').resize((28, 28))
    img = np.array(img)
    if np.mean(img) > 127:
        img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1).astype('float32')
    return img

def predict_digit(image, model):
    img = preprocess_image(image)
    prediction = model.predict(img)
    return np.argmax(prediction)

if "false_clicked" not in st.session_state:
    st.session_state.false_clicked = False

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False, width=150)

    model = load_model()
    predicted_digit = predict_digit(image, model)
    st.markdown(f"### 🧠 Model Prediction: `{predicted_digit}`")

    st.markdown("### ❓ Is this prediction correct?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ True"):
            st.session_state.false_clicked = False
            st.success("Thanks for your feedback! No image saved.")

    with col2:
        if st.button("❌ False"):
            st.session_state.false_clicked = True

    if st.session_state.false_clicked:
        correct_label = st.number_input("Enter the correct digit (0-9):", min_value=0, max_value=9, step=1)

        if st.button("Submit Correct Label"):
            os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            train_save_path = os.path.join(TRAIN_LABEL_DIR, f"{correct_label}_{timestamp}.png")
            image.save(train_save_path)

            st.success("Thanks! Image saved for retraining.")
            st.session_state.false_clicked = False

if st.button("Retrain model now with saved images"):
    with st.spinner("Retraining model, please wait..."):
        try:
            result = subprocess.run(['python3', 'retrain.py'], capture_output=True, text=True, check=True)
            st.success("Retraining completed successfully!")
            st.text(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error("Error during retraining.")
            st.text(e.stderr)
