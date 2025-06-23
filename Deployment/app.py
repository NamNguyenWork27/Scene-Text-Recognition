import streamlit as st
import requests
from PIL import Image
import io

st.title("OCR - Streamlit + FastAPI")
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)

    if st.button("Run OCR"):
        with st.spinner("Processing..."):
            files = {"file": (uploaded.name, uploaded.read(), uploaded.type)}
            response = requests.post("http://localhost:8000/ocr", files=files)

            if response.status_code == 200:
                
                preds = response.headers.get("X-Text", "")
                st.success("Recognized text:")
                for line in preds.split("|"):
                    st.write(f"• {line}")


                image_result = Image.open(io.BytesIO(response.content))
                st.image(image_result, caption="Annotated Image", use_column_width=True)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
