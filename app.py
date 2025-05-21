import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('best_model_CNN_Final.h5')

# Class labels for ASL (A-Z + special)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Set up the Streamlit UI
st.set_page_config(page_title="ASL Classifier", layout="centered")
st.title("ASL Alphabet Sign Classifier")
st.markdown("Upload a grayscale hand sign image (60x60) to classify the ASL letter.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((60, 60))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 60, 60, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence = float(np.max(prediction)) * 100

    # Display results
    st.markdown("---")
    st.subheader(f" Prediction: `{predicted_label}`")
    st.write(f"Confidence: **{confidence:.2f}%**")
