from ultralytics import YOLO
import cv2

# Load text detection model
model = YOLO("yolov8n.pt")  # temporary — we validate pipeline first

image_path = "test_img.jpg"

results = model(image_path)

for r in results:
    for box in r.boxes.xyxy.cpu().numpy():
        print("Box:", box.astype(int))