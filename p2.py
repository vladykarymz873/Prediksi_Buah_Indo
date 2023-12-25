import tensorflow as tf
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import requests

# Load the pre-trained model
model = tf.keras.models.load_model('fruit_model.h5')

# Fungsi untuk melakukan prediksi
def predict_fruits(img):
    # Praproses gambar
    img = img.resize((224, 224))  # Mengubah ukuran gambar
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi nilai piksel

    # Melakukan prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Get the class label
    labels = {
        0 : "apple",
        1 : "banana",
        2 : "bell pepper",
        3 : "cucumber",
        4 : "eggplant",
        5 : "grapes",
        6 : "kiwi",
        7 : "lemon",
        8 : "mango",
        9 : "orange",
        10 : "pear",
        11 : "pineapple",
        12 : "pomegranate",
        13 : "tomato",
        14 : "watermelon"
    } 
    predict_fruits_result = labels.get(predicted_class, 'Tidak Diketahui')
    probability = np.max(predictions[0])  # Menggunakan probabilitas tertinggi sebagai hasil

    return predict_fruits_result, probability

# Function to display image and prediction
def display_image_and_prediction(img, predict_fruits_result, probability):
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    st.success(f"Nama Buahnya adalah: {predict_fruits_result} dengan probabilitas: {probability:.1%}")

# Streamlit UI
st.title("Prediksi Buah-buahan Tropis dan Non-tropis")

# Option to upload image file
uploaded_file = st.file_uploader("Masukan Image Buah-buahan...", type=["jpg", "jpeg","png"])
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Make predictions when the user clicks the button
    if st.button("Predict"):
        predict_fruits_result, probability = predict_fruits(img)
        display_image_and_prediction(img, predict_fruits_result, probability)
