import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import time  # Import the time module

def load_tumor_classifier():
  """Loads the pre-trained CNN model for tumor classification."""
  try:
    model = load_model('cnn_model.h5', compile=False)
    return model
  except Exception as e:
    st.error(f"Failed to load model: {e}")
    return None

def process_image(image):
  """Preprocesses the uploaded image for model input."""
  try:
    img = image.resize((255, 255))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array
  except Exception as e:
    st.error(f"Failed to process image: {e}")
    return None

def predict_tumor_type(model, img_array, timeout=10):
  """Predicts the tumor type from the processed image with timeout."""
  if model is None or img_array is None:
    return None
  start_time = time.time()  # Use the imported time module
  while True:
    try:
      predictions = model.predict(img_array)
      return predictions
    except BrokenPipeError:
      if time.time() - start_time > timeout:
        st.error("An error occurred during prediction. Prediction timed out.")
        return None
      else:
        time.sleep(1)  # Wait for a short duration before retrying

def display_results(class_labels, predictions):
  """Displays the predicted tumor type probabilities."""
  st.write('### Predictions:')
  most_probable_class = class_labels[predictions[0].argmax()]
  most_probable_prob = predictions[0].max()
  st.write(f"Most probable tumor type: {most_probable_class} ({most_probable_prob:.2f})")
  for i, prob in enumerate(predictions[0]):
    st.write(f"Probability of {class_labels[i]}: {prob:.2f}")

# Load the model
model = load_tumor_classifier()

if model is not None:
  st.title('Brain MRI Tumor Classifier')
  st.write('Upload an MRI image of a brain tumor to classify the tumor type.')

  class_labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

  uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png"])

  if uploaded_file is not None:
    with st.spinner('Predicting...'):
      image = Image.open(uploaded_file)
      img_array = process_image(image)
      predictions = predict_tumor_type(model, img_array)

    if predictions is not None:
      st.image(image, caption='Uploaded Image', use_column_width=True)
      display_results(class_labels, predictions)
    else:
      st.error("An error occurred during prediction.")

  else:
    st.write("Please upload an image.")

st.write("**Disclaimer:** This is a demonstrative application and should not be used for medical diagnosis. Always consult a qualified medical professional for any medical concerns.")
