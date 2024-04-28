import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model

model = load_model('cnn_model.h5', compile=False)
lab = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

st.title("Birds Species Classification")

img_file = st.file_uploader("Choose an Image of Bird", type=["jpg", "png"])
if img_file is not None:
    st.image(img_file, use_column_width=False)
    save_image_path = './upload_images/' + img_file.name
    with open(save_image_path, "wb") as f:
        f.write(img_file.getbuffer())

    if st.button("Predict"):
        img = load_img(save_image_path, target_size=(255, 255, 3))
        img = img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, [0])
        answer = model.predict(img)
        y_class = answer.argmax(axis=-1)
        y = int(y_class)
        res = lab[y]
        st.success("Predicted Bird is: " + res)
