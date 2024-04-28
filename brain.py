import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

st.title('Brain MRI Tumor Classifier')
st.write('Upload an MRI image of a brain tumor to classify the tumor type.')
model = load_model('cnn_model.h5', compile=False)


class_labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        img = image.resize((255, 255))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        st.write('Processed Image Shape:', img_array.shape)
        st.write('Model Input Shape:', model.input_shape)

        try:
            predictions = model.predict(img_array)
            st.write('### Predictions:')
            for i, prob in enumerate(predictions[0]):
                st.write(f"Probability of {class_labels[i]}: {prob}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
