import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="VisionAssist AI",
    layout="wide"
)

st.title("👁️ VisionAssist AI")
st.write("Upload an image for object detection using YOLOv8.")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)

    results = model(image_np)

    annotated = results[0].plot()

    st.image(
        annotated,
        channels="BGR",
        caption="Detection Result",
        width="stretch"
    )

    st.subheader("Detected Objects")

    if len(results[0].boxes) == 0:
        st.info("No objects detected.")
    else:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = results[0].names[cls]
            conf = float(box.conf[0])

            st.write(f"• {name} ({conf:.2f})")

    output_path = "outputs/result.jpg"

    cv2.imwrite(output_path, annotated)

    with open(output_path, "rb") as file:
        st.download_button(
            label="📥 Download Result",
            data=file,
            file_name="result.jpg",
            mime="image/jpeg"
        )
