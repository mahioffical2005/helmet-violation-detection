# detect.py - DL Helper utility for YOLO inference
import os
import cv2
import numpy as np
from ultralytics import YOLO
from logic import get_helmet_model, get_plate_model

def run_detection(image, conf_thres: float = 0.25):
    """
    Runs helmet detection on an image path or numpy array.
    Returns list of detected bounding boxes and labels.
    """
    model = get_helmet_model()
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image}")
    else:
        img = image

    results = model(img, conf=conf_thres, verbose=False)[0]
    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names.get(cls_id, str(cls_id))
        pts = [int(v) for v in box.xyxy[0]]
        detections.append({
            "label": label,
            "conf": conf,
            "box": pts
        })
    return detections

if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else "samples/sample_helmet.jpg"
    print(f"Running detection on {test_img}...")
    res = run_detection(test_img)
    for r in res:
        print(" ", r)