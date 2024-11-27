import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model(r"C:\Users\rumjhum\Desktop\Dogs and Cats Classification\model_rcat_dog.h5")

# Title and Instructions
st.title("🐾 Cat vs Dog Classifier 🐾")
st.write("Upload an image to classify it as a **Cat** or **Dog**!")

# Sidebar for file upload
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Ensure the temp folder exists
    temp_folder = "temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # Save the uploaded file locally
    file_path = os.path.join(temp_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the image
    def preprocess_image(img_path):
        test_image = image.load_img(img_path, target_size=(64, 64))
        test_image = image.img_to_array(test_image) / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        return test_image

    # Preprocess and predict
    test_image = preprocess_image(file_path)
    result = model.predict(test_image)

    # Interpret results
    if (result[0] < 0).all():
        st.success("The image is classified as **Cat** 🐱")
    else:
        st.success("The image is classified as **Dog** 🐶")

    # Clean up by removing the temporary file
    os.remove(file_path)
else:
    st.info("👆 Upload a file to start!")

# Add a footer for style
st.markdown(
    """
    ---
    Developed with ❤️ by **Priyankar**
    """
)
