import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from datetime import datetime


SAVE_FOLDER = "faces"
os.makedirs(SAVE_FOLDER, exist_ok=True)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")



def extract_largest_face(image, scale_factor=1.1, min_neighbors=5):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    if len(faces) == 0:
        return None

    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    face = img_cv[y:y+h, x:x+w]
    face = align_face(face)

    return face

def align_face(face_img):
    return cv2.resize(face_img, (200, 200))

def save_face(face_img):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = os.path.join(SAVE_FOLDER, f"face_{timestamp}.jpg")
    cv2.imwrite(save_path, face_img)
    return save_path

st.set_page_config(page_title="Certificate Face Extractor", layout="centered")

st.title("📜 Smart Certificate Face Extractor")
st.write("Upload certificate images and automatically extract the largest detected face (person's photo).")

# Sidebar settings
st.sidebar.header("⚙ Detection Settings")
scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.5, 1.1, 0.01)
min_neighbors = st.sidebar.slider("Min Neighbors", 3, 10, 5, 1)

uploaded_files = st.file_uploader("Upload Certificate Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"Processing: {uploaded_file.name}")

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Certificate", use_container_width=True)

        face = extract_largest_face(image, scale_factor=scale_factor, min_neighbors=min_neighbors)

        if face is not None:
            st.success("✅ Face detected and extracted.")
            st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption="Extracted Face", width=200)
            save_path = save_face(face)
            st.write(f"💾 Saved to: `{save_path}`")

            st.download_button(
                label="⬇ Download Face",
                data=cv2.imencode('.jpg', face)[1].tobytes(),
                file_name=f"extracted_face_{idx}.jpg",
                mime="image/jpeg",
                key=f"download_button_{idx}"  
            )
        else:
            st.error("❌ No face detected in this image.")

