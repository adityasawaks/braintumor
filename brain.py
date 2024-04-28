import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Brain Tumor MRI Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    st.write("Image uploaded successfully.")

    # Convert image to array
    img = image.resize((255, 255))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    print(img_array)

    # Load the pre-trained model
    st.write("Loading pre-trained model...")
    model = tf.keras.models.load_model("cnn_model.h5")
    st.write("Model loaded successfully.")

    # Define the class labels
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Preprocess the image
    st.write("Preprocessing the image...")
    try:
        # Make predictions
        predictions = model.predict(img_array)
        st.write("Predictions:")
        for i, prob in enumerate(predictions[0]):
            st.write(f"Probability of {class_labels[i]}: {prob}")
    except Exception as e:
        st.write(f"An error occurred: {e}")
