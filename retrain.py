import numpy as np
import tensorflow as tf
from PIL import Image
import os

MODEL_PATH = 'digitmodel.h5'
TRAIN_LABEL_DIR = "train_label"

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
        print(f"No training label directory found at {train_label_dir}")
        return None, None
    for filename in os.listdir(train_label_dir):
        if filename.endswith('.png'):
            try:
                label = int(filename.split('_')[0])
            except:
                print(f"Skipping file with invalid label format: {filename}")
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

def retrain_model_on_new_data_only(model_path=MODEL_PATH, train_label_dir=TRAIN_LABEL_DIR, epochs=10):
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("Loading new labeled images for retraining...")
    X_new, y_new = load_train_data_from_folder(train_label_dir)

    if X_new is None or len(X_new) == 0:
        print("No new labeled images found. Aborting retraining.")
        return

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Starting training on {len(X_new)} new images for {epochs} epochs...")
    model.fit(X_new, y_new, epochs=epochs, batch_size=16, verbose=1)

    print(f"Saving retrained model to {model_path} ...")
    model.save(model_path)
    print("Retraining complete and model saved.")

if __name__ == "__main__":
    retrain_model_on_new_data_only()
