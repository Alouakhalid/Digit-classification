# 🧠 Handwritten Digit Recognition with Self-Improving Feedback

This project is a **web app** that recognizes handwritten digits using a trained deep learning model built with **TensorFlow**, and deployed via **Streamlit**.

The app allows users to upload an image of a digit (0–9). The model predicts the digit, and if the prediction is wrong, the image is saved in a folder (named after the correct label) for **future retraining**, making the model smarter over time.

---

## 🚀 Features

- ✅ Predicts handwritten digits (0–9) from uploaded images.
- 📁 If prediction is **wrong**, the image is saved under a folder `train_LABEL/` for retraining.
- 🧠 Trains on new data collected from incorrect predictions.
- 🖼️ Streamlit UI for easy interaction.

---

## 📸 Demo

[Add screenshots or link to demo video here]

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Streamlit
- NumPy
- PIL

---

## 📂 Project Structure
digit_recognition/
├── app.py                 # Streamlit app 
├── digitmodel.h5               # Pre-trained TensorFlow model 
├── main.ipynb             # Prediction logic 
├── retrain.py             # Optional retraining script 
├── train_label          # Folder for incorrectly predicted images -
├── test.ipynb               # Image preprocessing utilities 
└── requirements.txt       # Project dependencies 
