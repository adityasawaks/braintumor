import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the Keras model
# @st.cache(allow_output_mutation=True)
# def load_model():

# Define class labels
class_labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

# Main function to make predictions
def predict(image):
    model = load_model()
    img = image.resize((255, 255))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    tf.keras.models.load_model('cnn_model.h5')
    predictions = model.predict(img_array)
    return predictions

# Streamlit app
def main():
    st.title('Brain Tumor Classifier')
    st.write('Upload an MRI image of a brain tumor to classify the tumor type.')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        predictions = predict(image)

        st.write('### Predictions:')
        for i, prob in enumerate(predictions[0]):
            st.write(f"Probability of {class_labels[i]}: {prob}")

if __name__ == '__main__':
    main()
