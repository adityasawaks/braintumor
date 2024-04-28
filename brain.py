import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model without compiling
model = tf.keras.models.load_model('cnn_model.h5', compile=False)

# Compile the model with specified optimizer, loss, and metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Streamlit app
def main():
    st.title("Brain Tumor MRI Image Classification")

    # Image path
    image_path = '/kaggle/input/brain-tumor-mri-dataset/Testing/pituitary/Te-piTr_0004.jpg'
    image = Image.open(image_path)

    # Display the image
    st.image(image, caption='Test Image', use_column_width=True)
    
    # Preprocess the image
    image = image.resize((255, 255))
    image = np.array(image) / 255.0
    img_array = np.expand_dims(image, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_class_index]

    st.write("Prediction:", predicted_label)

if __name__ == '__main__':
    main()
