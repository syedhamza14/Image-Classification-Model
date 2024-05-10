import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Load the model
model = load_model('/Users/hamza/Desktop/Image_Classification/Image_classify.keras')

# Define the categories
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Set image dimensions
img_height = 180
img_width = 180

# UI header
st.header('Image Classification Model')

# File uploader to select multiple images
uploaded_files = st.file_uploader("Upload image files...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Perform prediction for each uploaded image
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # Read and preprocess the uploaded image
            image = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
            img_arr = tf.keras.utils.img_to_array(image)
            img_bat = np.expand_dims(img_arr, axis=0)

            # Make prediction
            predict = model.predict(img_bat)
            score = tf.nn.softmax(predict)

            # Display the image and prediction results
            st.image(image, width=200)
            st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
            st.write('With accuracy of {:.2f}%'.format(np.max(score) * 100))
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

