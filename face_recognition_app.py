import streamlit as st
import joblib
import numpy as np
from PIL import Image
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer
from Extract_Face import extract_face

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="Face Recognition App 🤖",
    page_icon="🧠",
    layout="centered"
)
list=["Angelina_Jolie","Emma Watson","Leonardo_DiCaprio","Scarlett Johansson","Will_Smith","none"]
# -------------------------------
# Title and instructions
# -------------------------------
st.markdown("""
    <h1 style='text-align: center; color: #2E8B57;'>🌟 Face Recognition Interface 🌟</h1>
    <p style='text-align: center; color: #555;'>Upload an image to recognize the person using the trained model</p>
    <hr style='border: 1px solid #2E8B57;'>
    """, unsafe_allow_html=True)

# -------------------------------
# Load model, embedder and normalizer with caching
# -------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("knn_model.joblib")  # Your trained KNN model path
    embedder = FaceNet()
    normalizer = Normalizer(norm='l2')
    return model, embedder, normalizer

model, embedder, normalizer = load_resources()

# -------------------------------
# Extract embedding from image
# -------------------------------
def get_face_embedding(image: np.ndarray):
    faces = embedder.extract(image, threshold=0.95)
    if faces:
        embedding = faces[0]['embedding']
        embedding_norm = normalizer.transform([embedding])
        return embedding_norm
    else:
        return None

# -------------------------------
# Upload file and predict
# -------------------------------
uploaded_file = st.file_uploader("📷 Upload a face image", type=['jpg', 'jpeg', 'png'])

import tempfile

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🧑 Input Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    with st.spinner("Extracting face and predicting..."):
        face_pixels = extract_face(tmp_path)
        if face_pixels is not None:
            embedding = get_face_embedding(face_pixels)
            if embedding is not None:
                # Get prediction and probabilities
                prediction = model.predict(embedding)
                probabilities = model.predict_proba(embedding)[0]
                max_prob = max(probabilities)  # Get the highest probability
                if max_prob >= 0.9:  # Check if it meets the 70% threshold
                    st.success(f"🎉 Predicted Identity: **{list[prediction[0]]}** (Confidence: {max_prob:.2%})")
                else:
                    st.warning("😕 Prediction confidence below 90%. No reliable identity match.")
            else:
                st.warning("😕 Could not generate embedding for the detected face.")
        else:
            st.warning("😕 No face detected in the image or face too small.")