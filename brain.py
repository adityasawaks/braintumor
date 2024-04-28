import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('cnn_model.h5')

# Define class labels
class_labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    image = image.resize((255, 255))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the image
    image = image / 255.0
    # Expand dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
def main():
    st.title("CNN Image Classification App")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        processed_image = preprocess_image(image)
        # Predict
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]
        st.write("Prediction:", predicted_label)

if __name__ == '__main__':
    main()
