import streamlit as st
import tensorflow as tf
from PIL import Image
import time

# Load your pre-trained model
print("Loading pre-trained model...")
model = tf.keras.models.load_model("cnn_model.h5")
print("Model loaded successfully.")

# Define the class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("Brain Tumor MRI Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    print("Image uploaded successfully.")

    # Preprocess the image
    print("Preprocessing the image...")
    img = image.resize((255, 255))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    print("Image preprocessing completed.")

    try:
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(img_array)
        print("Predictions made successfully.")

        st.write("Predictions:")
        for i, prob in enumerate(predictions[0]):
            st.write(f"Probability of {class_labels[i]}: {prob}")

    except Exception as e:
        st.write(f"An error occurred: {e}")
        time.sleep(5)  # Adding a delay to keep the app alive for 5 seconds

