import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io

st.set_page_config(page_title="VisionAssist AI", layout="wide")

st.title("👁️ VisionAssist AI")
st.write("Upload an image for object detection using YOLOv8.")

# Load model
model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    with st.spinner("Detecting objects..."):
        results = model(image_np)

    annotated = results[0].plot()

    st.image(
        annotated,
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

    # Save image using Pillow (NO OpenCV)
    img = Image.fromarray(annotated[:, :, ::-1])

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    st.download_button(
        "📥 Download Result",
        data=buffer,
        file_name="result.jpg",
        mime="image/jpeg"
    )
