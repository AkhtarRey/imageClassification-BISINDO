import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model without compilation
model = load_model('./model.h5', compile=False)

# Recompile the model with appropriate loss and optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define class names (A-Z)
class_names = [chr(i) for i in range(65, 91)]  # A-Z

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

# Function to validate if the prediction is a sign language alphabet
def is_valid_prediction(prediction, threshold=0.5):
    confidence = np.max(prediction)
    return confidence > threshold

# Streamlit interface
st.title("BISINDO Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)
    
    st.write("Classifying...")
    
    image = preprocess_image(image)
    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    
    if is_valid_prediction(predictions):
        st.write(f"Predicted Class: {class_names[predicted_class_idx]} (Index: {predicted_class_idx})")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.write("The uploaded image is not a valid sign language alphabet.")

# Display the image below the title
st.image("./bisindo.jpg", caption='BISINDO', width=300)