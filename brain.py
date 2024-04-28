import streamlit as st
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np

model = tf.keras.models.load_model('path_to_your_trained_model')

model = load_model('cnn_model.h5', compile=False)
lab = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

st.title("Brain Tumor Type Classification")

img_file = st.file_uploader("Choose an Image of Brain MRI", type=["jpg", "png"])
if img_file is not None:
    st.image(img_file, use_column_width=False)

    if st.button("Predict"):
        img = Image.open(img_file)
        img = img.resize((255, 255))  # Resize image
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        tumor_type = lab[predicted_class]
        st.success("Predicted Tumor Type: " + tumor_type)
