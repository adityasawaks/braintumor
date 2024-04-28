import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
try:
    model = tf.keras.models.load_model('cnn_model.h5', compile=False)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"An error occurred while loading the model: {str(e)}")

# Compile the model with specified optimizer, loss, and metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Streamlit app
def main():
    st.title("Brain Tumor MRI Image Classification")

    # File uploader for image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open the uploaded image
        image = Image.open(uploaded_image)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        image = image.resize((255, 255))
        image = np.array(image) / 255.0
        img_array = np.expand_dims(image, axis=0)
        
        # Debugging: Print shape of img_array
        st.write("Image Array Shape:", img_array.shape)

        try:
            # Make predictions
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_label = class_labels[predicted_class_index]

            st.write("Prediction:", predicted_label)
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    main()

