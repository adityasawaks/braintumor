import streamlit as st
import tensorflow as tf
from PIL import Image

st.title("Brain Tumor MRI Classification")

# Define the class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Load the model
    try:
        model = tf.keras.models.load_model("cnn_model.h5")
    except OSError:
        st.error("Error: Unable to load the model.")
    else:
        # Preprocess the image
        img = image.resize((255, 255))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Make predictions
        predictions = model.predict(img_array)
        
        # Display the predictions
        st.write("Predictions:")
        predicted_label_index = tf.argmax(predictions[0])
        predicted_label = class_labels[predicted_label_index]
        st.write(f"Predicted class: {predicted_label}")
