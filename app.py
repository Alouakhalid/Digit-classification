import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from datetime import datetime

st.set_page_config(
    page_title="Digit Classifier with Feedback & Retraining",
    layout="centered",
    page_icon="🧠"  # Brain emoji favicon
)

st.title("🔢 Handwritten Digit Classifier with Feedback & Retraining")
st.markdown(
    "Upload a **28x28 grayscale image** of a digit (0–9). The model will predict the digit.\n"
    "If the prediction is wrong, provide the correct label and save it for retraining."
)

MODEL_PATH = 'digitmodel.h5'
SAVED_IMAGES_DIR = "saved_images"
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

def preprocess_image_for_training(image):
    img = image.convert('L').resize((28, 28))
    img = np.array(img)
    if np.mean(img) > 127:
        img = 255 - img
    img = img / 255.0
    img = img.reshape(28, 28, 1).astype('float32')
    return img

def load_train_data_from_folder(train_label_dir=TRAIN_LABEL_DIR):
    images = []
    labels = []
    if not os.path.exists(train_label_dir):
        return None, None
    for filename in os.listdir(train_label_dir):
        if filename.endswith('.png'):
            try:
                label = int(filename.split('_')[0])
            except:
                continue
            path = os.path.join(train_label_dir, filename)
            img = Image.open(path)
            img = preprocess_image_for_training(img)
            images.append(img)
            labels.append(label)
    if images:
        return np.array(images), np.array(labels)
    else:
        return None, None

def load_mnist_data():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    return x_train, y_train

def retrain_model_with_mnist_and_new_data(model_path=MODEL_PATH, train_label_dir=TRAIN_LABEL_DIR, epochs=10):
    model = tf.keras.models.load_model(model_path)

    # Load original MNIST training data
    x_train, y_train = load_mnist_data()

    # Load newly labeled images
    X_new, y_new = load_train_data_from_folder(train_label_dir)

    if X_new is not None:
        # Combine datasets
        X_combined = np.concatenate((x_train, X_new), axis=0)
        y_combined = np.concatenate((y_train, y_new), axis=0)
    else:
        st.warning("No new labeled images found for retraining, training only on MNIST.")
        X_combined, y_combined = x_train, y_train

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    with st.spinner("Retraining model... This may take some time."):
        model.fit(X_combined, y_combined, epochs=epochs, batch_size=64, verbose=1)

    model.save(model_path)
    st.success("Model retrained and saved!")

    return model

# --- App logic starts here ---

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if "false_clicked" not in st.session_state:
    st.session_state.false_clicked = False

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
            os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)
            os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            raw_save_path = os.path.join(SAVED_IMAGES_DIR, f"wrong_pred_{timestamp}.png")
            image.save(raw_save_path)

            train_save_path = os.path.join(TRAIN_LABEL_DIR, f"{correct_label}_{timestamp}.png")
            image.save(train_save_path)

            st.success("Thanks! Images saved for retraining.")

            st.session_state.false_clicked = False

st.markdown("---")
if st.button("Retrain model now with saved images and MNIST"):
    retrain_model_with_mnist_and_new_data()
