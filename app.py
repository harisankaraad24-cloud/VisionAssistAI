import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="VisionAssist AI", layout="wide")

st.title("👁️ VisionAssist AI")
st.write("Upload an image for real-time object detection using YOLOv8.")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image_np = np.array(image)

    results = model(image_np)

    annotated = results[0].plot()

    st.image(
        annotated,
        channels="BGR",
        caption="Detection Result",
        use_container_width=True
    )

    st.subheader("Detected Objects")

    for box in results[0].boxes:

        cls = int(box.cls[0])

        name = results[0].names[cls]

        conf = float(box.conf[0])

        st.write(f"• {name} ({conf:.2f})")

    cv2.imwrite("outputs/result.jpg", annotated)

    with open("outputs/result.jpg", "rb") as file:

        st.download_button(
            "Download Result",
            file,
            file_name="result.jpg"
        )
