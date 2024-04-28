import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def preprocess_mri(image):
    img = image.resize((255, 255))  # Resize the image
    img_array = img_to_array(img) / 255.0  # Convert to array and normalize
    return img_array

def predict_mri(model, image):
    st.write('Processed Image Shape:', image.shape)
    st.write('Model Input Shape:', model.input_shape)
    try:
        predictions = model.predict(image)
        return predictions
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def run():
    st.title('Brain MRI Tumor Classifier')
    st.write('Upload an MRI image of a brain tumor to classify the tumor type.')

    try:
        model = load_model('cnn_model.h5', compile=False)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return

    class_labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Predict"):
            processed_image = preprocess_mri(image)
            processed_image = processed_image.reshape(1, 255, 255, 3)  # Add batch dimension
            predictions = predict_mri(model, processed_image)
            if predictions is not None:
                st.write('### Predictions:')
                for i, prob in enumerate(predictions[0]):
                    st.write(f"Probability of {class_labels[i]}: {prob}")

if __name__ == '__main__':
    run()
