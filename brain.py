import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_mri(image):
    img = image.resize((255, 255))  # Resize the image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

def predict_mri(model, image):
    predictions = model.predict(image)
    return predictions

def run():
    st.title('Brain MRI Tumor Classifier')
    st.write('Upload an MRI image of a brain tumor to classify the tumor type.')

    model = load_model('cnn_model.h5', compile=False)
    class_labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Predict"):
            processed_image = preprocess_mri(image)
            predictions = predict_mri(model, processed_image)
            st.write('### Predictions:')
            for i, prob in enumerate(predictions[0]):
                st.write(f"Probability of {class_labels[i]}: {prob}")

if __name__ == '__main__':
    run()

