from ultralytics import YOLO

# Load the YOLOv8 Nano model
model = YOLO("yolov8n.pt")

def detect(frame):
    """
    Detect objects in a frame.
    """
    results = model(frame)
    return results
