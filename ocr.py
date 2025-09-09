import streamlit as st
import easyocr
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import cv2
from pdf2image import convert_from_bytes

st.title("📄 Multi-language EasyOCR with Preprocessing")

LANG_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "Chinese": "ch_sim",
    "Arabic": "ar",
    "Russian": "ru",
}

selected_langs = st.multiselect(
    "Select languages for OCR",
    options=list(LANG_OPTIONS.keys()),
    default=["English"]
)
languages = [LANG_OPTIONS[lang] for lang in selected_langs]

# File uploader supports images and PDFs
uploaded_file = st.file_uploader(
    "Upload an image or PDF file",
    type=["png", "jpg", "jpeg", "pdf"]
)

st.sidebar.header("Image Preprocessing Options")
grayscale = st.sidebar.checkbox("Convert to Grayscale", value=False)
threshold = st.sidebar.checkbox("Apply Thresholding (Binarization)", value=False)
enhance_contrast = st.sidebar.slider("Contrast Enhancement", 1.0, 3.0, 1.0, step=0.1)

reader = easyocr.Reader(languages)

def preprocess_image(image):
    if grayscale:
        image = image.convert("L").convert("RGB")
    if enhance_contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(enhance_contrast)
    if threshold:
        gray = np.array(image.convert("L"))
        _, thresh_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = Image.fromarray(thresh_img).convert("RGB")
    return image

def draw_boxes(image, results):
    draw = ImageDraw.Draw(image)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        pass
    for (bbox, text, prob) in results:
        bbox = [tuple(point) for point in bbox]
        draw.line(bbox + [bbox[0]], width=2, fill="red")
        draw.text(bbox[0], text, fill="red", font=font)
    return image

if uploaded_file is not None:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        # Convert PDF pages to images
        try:
            pages = convert_from_bytes(uploaded_file.read(),poppler_path=r"C:\poppler-24.08.0\Library\bin")
        except Exception as e:
            st.error(f"Error converting PDF: {e}")
            st.stop()

        all_text = ""
        for i, page in enumerate(pages):
            st.write(f"--- Page {i+1} ---")
            image = preprocess_image(page)
            st.image(image, caption=f"Page {i+1} - Preprocessed", use_container_width=True)

            image_np = np.array(image)
            with st.spinner(f"Performing OCR on page {i+1}..."):
                results = reader.readtext(image_np)

            extracted_text = "\n".join([res[1] for res in results])
            all_text += f"\n\n--- Page {i+1} ---\n{extracted_text}"

            st.subheader("Extracted Text:")
            st.text_area(f"Text Output Page {i+1}", extracted_text, height=200)

            image_with_boxes = draw_boxes(image.copy(), results)
            st.image(image_with_boxes, caption=f"Page {i+1} with Bounding Boxes", use_container_width=True)

        st.download_button(
            label="Download All Text as .txt",
            data=all_text,
            file_name="ocr_output_all_pages.txt",
            mime="text/plain"
        )

    else:
        # Handle image file
        image = Image.open(uploaded_file).convert("RGB")
        image = preprocess_image(image)
        st.image(image, caption="Preprocessed Image", use_container_width=True)

        image_np = np.array(image)
        with st.spinner("Performing OCR..."):
            results = reader.readtext(image_np)

        extracted_text = "\n".join([res[1] for res in results])
        st.subheader("Extracted Text:")
        st.text_area("Text Output", extracted_text, height=200)

        image_with_boxes = draw_boxes(image.copy(), results)
        st.image(image_with_boxes, caption="Image with Bounding Boxes", use_container_width=True)

        st.download_button(
            label="Download Text as .txt",
            data=extracted_text,
            file_name="ocr_output.txt",
            mime="text/plain"
        )
