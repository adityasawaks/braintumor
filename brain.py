import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_model_and_labels(model_path):
  """Loads the pre-trained model and class labels (assuming a built-in dictionary)."""
  # Load the model
  model = load_model(cnn_model.h5)

  # Define class labels directly within the function
  class_labels = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

  return model, class_labels

def preprocess_image(img, target_size=(255, 255), normalize=True):
  """Preprocesses the MRI image for the model."""
  img = Image.open(img)
  img = img.resize(target_size)
  img_array = tf.keras.preprocessing.image.img_to_array(img)

  # Normalize pixel values (adjust based on model requirements)
  if normalize:
    img_array = img_array / 255.0

  # Additional preprocessing steps specific to MRI analysis (e.g., skull stripping)
  # may be required depending on the model

  img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
  return img_array

def predict_mri(model, img_array, class_labels):
  """Makes predictions on the preprocessed MRI image."""
  predictions = model.predict(img_array)
  predicted_class = np.argmax(predictions[0])
  predicted_label = class_labels[list(class_labels.keys())[predicted_class]]
  return predicted_label

def run():
  """Defines the Streamlit application logic."""
  st.title("MRI Brain Tumor Classification")

  # Model path input (modify as needed)
  model_path = st.text_input("Enter the path to your pre-trained model (.h5)", "/path/to/model.h5")

  # Load the model and labels (built-in dictionary)
  try:
    model, class_labels = load_model_and_labels(model_path)
    st.success("Model and labels loaded successfully!")
  except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    return

  # Image upload and prediction
  img_file = st.file_uploader("Choose an MRI Image", type=["jpg", "png"])
  if img_file is not None:
    st.image(img_file, use_column_width=False)
    save_image_path = './upload_images/' + img_file.name
    with open(save_image_path, "wb") as f:
      f.write(img_file.getbuffer())

    if st.button("Predict"):
      try:
        img_array = preprocess_image(save_image_path)
        predicted_result = predict_mri(model, img_array, class_labels)
        st.success(f"Predicted Brain Tumor: {predicted_result}")
      except Exception as e:
        st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
  run()
