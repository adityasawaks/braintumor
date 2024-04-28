import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except OSError:
        st.error("Error: Unable to load the model.")
        return None

def preprocess_image(image):
    img = image.resize((255, 255))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_class(model, img_array, class_labels):
    try:
        predictions = model.predict(img_array)
        predicted_label_index = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_label_index]
        st.write("Predicted class:", predicted_label)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

def main():
    st.title("Brain Tumor MRI Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)
        st.write("Image uploaded successfully.")

        model_path = "cnn_model.h5"
        model = load_model(model_path)

        if model is not None:
            class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
            img_array = preprocess_image(image)
            predict_class(model, img_array, class_labels)

if __name__ == "__main__":
    main()
