import streamlit as st
import tensorflow as tf
from PIL import Image

st.title('Brain Tumor Image Classifier')

try:
    model = tf.keras.models.load_model('cnn_model.h5', compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    st.write("Model loaded and compiled successfully!")
except Exception as e:
    st.error(f"An error occurred while loading the model: {str(e)}")

# Define the class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    # Open and preprocess the image
    image = Image.open(uploaded_image)
    img = image.resize((255, 255))
    
    # Normalize the image
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)
    
    try:
        # Make predictions
        predictions = model.predict(img_array)

        # Display predictions
        st.subheader("Prediction Results:")
        for i, prob in enumerate(predictions[0]):
            st.write(f"Probability of {class_labels[i]}: {prob}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
