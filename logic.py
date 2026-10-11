import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Configuration & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HELMET_MODEL_LOCAL = os.path.join(MODELS_DIR, "helmet_model.pt")
PLATE_MODEL_LOCAL = os.path.join(MODELS_DIR, "plate_model.pt")

_helmet_model = None
_plate_model = None
_ocr_reader = None

def get_helmet_model():
    """Loads helmet YOLOv8 model from local folder or Hugging Face Hub."""
    global _helmet_model
    if _helmet_model is not None:
        return _helmet_model

    if os.path.exists(HELMET_MODEL_LOCAL):
        path = HELMET_MODEL_LOCAL
    else:
        print("[DL Logic] Downloading helmet model from Hugging Face Hub...")
        path = hf_hub_download(repo_id="iam-tsr/yolov8n-helmet-detection", filename="best.pt")
    
    _helmet_model = YOLO(path)
    return _helmet_model

def get_plate_model():
    """Loads license plate YOLOv8 model from local folder or Hugging Face Hub."""
    global _plate_model
    if _plate_model is not None:
        return _plate_model

    if os.path.exists(PLATE_MODEL_LOCAL):
        path = PLATE_MODEL_LOCAL
    else:
        print("[DL Logic] Downloading license plate model from Hugging Face Hub...")
        path = hf_hub_download(repo_id="yasirfaizahmed/license-plate-object-detection", filename="best.pt")
    
    _plate_model = YOLO(path)
    return _plate_model

def get_ocr_reader():
    """Initializes EasyOCR reader safely."""
    global _ocr_reader
    if _ocr_reader is not None:
        return _ocr_reader
    
    try:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    except Exception as e:
        print(f"[DL Logic] Warning: EasyOCR reader initialization error: {e}")
        _ocr_reader = None
    return _ocr_reader

INDIAN_STATE_CODES = [
    "AN","AP","AR","AS","BR","CH","CG","DD","DL","DN","GA","GJ","HP",
    "HR","JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL",
    "OD","OR","PB","PY","RJ","SK","TN","TR","TS","UK","UP","WB"
]

def format_indian_plate(tokens):
    """
    Reconstructs standard Indian vehicle plate format (e.g. TN 09 CF 7722).
    State Code (2) + District (2) + Series (1-2) + Number (4).
    Handles single-line and multi-line plate layouts.
    """
    cleaned_tokens = [re.sub(r'[^A-Z0-9]', '', t.upper()) for t in tokens if t.strip()]
    
    # 1. Look for token that starts with state code
    state_idx = -1
    for i, t in enumerate(cleaned_tokens):
        for code in INDIAN_STATE_CODES:
            if t.startswith(code):
                state_idx = i
                break
        if state_idx != -1:
            break

    # If state code token is found later (e.g. bottom line read first), reorder so state line is first
    if state_idx > 0:
        reordered = cleaned_tokens[state_idx:] + cleaned_tokens[:state_idx]
    else:
        reordered = cleaned_tokens

    combined = "".join(reordered)

    # Search for state code inside combined
    for code in INDIAN_STATE_CODES:
        pos = combined.find(code)
        if pos != -1:
            # Reconstruct string starting from state code
            candidate = combined[pos:] + combined[:pos]
            # Pattern: State(2) + Num(1-2) + Letters(1-3) + Num(3-4)
            m = re.match(r'^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{3,4})', candidate)
            if m:
                return f"{m.group(1)} {m.group(2).zfill(2)} {m.group(3)} {m.group(4)}"
            
            # Pattern: State(2) + Letters(1-3) + Num(3-4)
            m_short = re.match(r'^([A-Z]{2})([A-Z]{1,3})([0-9]{3,4})', candidate)
            if m_short:
                return f"{m_short.group(1)} {m_short.group(2)} {m_short.group(3)}"
            
            # If at least state code followed by something
            return candidate

    return " ".join(cleaned_tokens)

def clean_plate_text(text: str) -> str:
    """Cleans raw OCR text into standard uppercase alphanumeric registration number format."""
    if not text:
        return ""
    clean = re.sub(r'[^A-Za-z0-9]', ' ', text).strip()
    clean = re.sub(r'\s+', ' ', clean)
    return clean.upper()

def draw_bounding_box(image: np.ndarray, box, label: str, color: tuple, conf: float = None):
    """Draws a clean bounding box with a high contrast label banner."""
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    thickness = max(2, int(min(h, w) * 0.004))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    display_text = f"{label} ({int(conf*100)}%)" if conf is not None else label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w, h) * 0.0009)
    font_thickness = max(1, int(font_scale * 2))
    
    (txt_w, txt_h), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)
    banner_y1 = max(0, y1 - txt_h - 10)
    banner_y2 = y1
    banner_x2 = min(w, x1 + txt_w + 10)
    
    cv2.rectangle(image, (x1, banner_y1), (banner_x2, banner_y2), color, -1)
    cv2.putText(
        image,
        display_text,
        (x1 + 5, banner_y2 - 5),
        font,
        font_scale,
        (255, 255, 255) if sum(color) < 400 else (0, 0, 0),
        font_thickness,
        cv2.LINE_AA
    )

def detect_violation(image_input, conf_threshold: float = 0.20):
    """
    Main detection pipeline:
    1. Runs YOLOv8 Helmet Detection ('With Helmet', 'Without Helmet')
    2. Runs YOLOv8 License Plate Detection
    3. Crops plate and reads text via EasyOCR
    4. Computes violation rules:
       - If Without Helmet detected: Status = 'No Helmet Detected ❌', Fine = 'Fine: ₹1000 🚨'
       - If With Helmet detected: Status = 'Helmet Detected ✅', Fine = 'No Fine (₹0) ✨'
    """
    if image_input is None:
        return None, "No Image Uploaded", "₹0", "N/A", "Please upload a photo of a rider or traffic scene."

    # Convert to OpenCV BGR
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image_input.copy()
    else:
        from PIL import Image
        if isinstance(image_input, Image.Image):
            img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            return None, "Error", "Error", "Error", "Unsupported image format."

    if img_bgr is None:
        return None, "Error", "Error", "Error", "Could not read image."

    annotated = img_bgr.copy()
    h_img, w_img = img_bgr.shape[:2]

    # --- 1. Helmet Detection ---
    helmet_model = get_helmet_model()
    # Try primary threshold first, fallback to slightly lower if nothing detected
    h_results = helmet_model(img_bgr, conf=conf_threshold, verbose=False)[0]
    if len(h_results.boxes) == 0:
        h_results = helmet_model(img_bgr, conf=0.15, verbose=False)[0]

    helmets = []
    no_helmets = []

    for box_obj in h_results.boxes:
        cls_id = int(box_obj.cls[0])
        conf = float(box_obj.conf[0])
        box = [int(v) for v in box_obj.xyxy[0]]
        cls_name = helmet_model.names.get(cls_id, f"Class {cls_id}")

        if "without" in cls_name.lower() or "no" in cls_name.lower():
            no_helmets.append({"box": box, "conf": conf, "label": cls_name})
        else:
            helmets.append({"box": box, "conf": conf, "label": cls_name})

    # Sort by confidence
    helmets.sort(key=lambda x: -x["conf"])
    no_helmets.sort(key=lambda x: -x["conf"])

    # Draw helmet / no-helmet bounding boxes
    for h in helmets:
        draw_bounding_box(annotated, h["box"], "Helmet Detected", (46, 204, 113), h["conf"])  # Green

    for nh in no_helmets:
        draw_bounding_box(annotated, nh["box"], "No Helmet", (41, 41, 230), nh["conf"])  # Red

    # --- 2. License Plate Detection & OCR ---
    plate_model = get_plate_model()
    reader = get_ocr_reader()
    p_results = plate_model(img_bgr, conf=0.12, verbose=False)[0]

    detected_plates = []

    for box_obj in p_results.boxes:
        box = [int(v) for v in box_obj.xyxy[0]]
        conf = float(box_obj.conf[0])
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        crop = img_bgr[y1:y2, x1:x2]
        plate_str = ""

        if crop.size > 0 and reader is not None:
            try:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # Resize if crop is too small for OCR
                if gray_crop.shape[0] < 60:
                    scale = 80 / max(1, gray_crop.shape[0])
                    gray_crop = cv2.resize(gray_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                
                ocr_out = reader.readtext(gray_crop)
                # Filter valid text items
                valid_items = [r for r in ocr_out if r[2] >= 0.10 and len(r[1].strip()) > 0]
                
                # Sort spatially: top-to-bottom, then left-to-right
                # r[0] is list of 4 bbox points [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                def get_sort_key(item):
                    box_pts = item[0]
                    y_center = (box_pts[0][1] + box_pts[2][1]) / 2.0
                    x_center = (box_pts[0][0] + box_pts[2][0]) / 2.0
                    # Group by horizontal line bands (30px)
                    line_band = int(y_center / 30.0)
                    return (line_band, x_center)
                
                valid_items.sort(key=get_sort_key)
                tokens = [r[1] for r in valid_items]
                plate_str = format_indian_plate(tokens)
            except Exception as e:
                print(f"[DL Logic] OCR exception: {e}")

        detected_plates.append({
            "box": box,
            "conf": conf,
            "text": plate_str
        })

    # Fallback OCR: scan lower half of image if plate model didn't yield text
    if (not detected_plates or not any(p["text"] for p in detected_plates)) and reader is not None:
        try:
            bottom_half = img_bgr[int(h_img * 0.4):h_img, :]
            ocr_all = reader.readtext(bottom_half)
            for res in ocr_all:
                cleaned = clean_plate_text(res[1])
                if len(cleaned) >= 4 and any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned):
                    detected_plates.append({
                        "box": [0, int(h_img * 0.4), w_img, h_img],
                        "conf": float(res[2]),
                        "text": cleaned
                    })
                    break
        except Exception:
            pass

    # Extract final plate text
    final_plate_text = "Not Visible in Image"
    for p in detected_plates:
        if p["text"]:
            final_plate_text = p["text"]
            draw_bounding_box(annotated, p["box"], f"Plate: {final_plate_text}", (0, 215, 255), p["conf"])
            break
        else:
            draw_bounding_box(annotated, p["box"], "License Plate", (0, 215, 255), p["conf"])

    # --- 3. Violation & Fine Logic ---
    # Decisive rule:
    # If any without-helmet detected with higher confidence than helmet or no helmets:
    if len(no_helmets) > 0 and (len(helmets) == 0 or no_helmets[0]["conf"] >= helmets[0]["conf"] * 0.85):
        helmet_status = "No Helmet Detected ❌"
        fine_text = "Fine: ₹1000 🚨"
        violation_flag = True
        conf_score = int(no_helmets[0]["conf"] * 100)
    elif len(helmets) > 0:
        helmet_status = "Helmet Detected ✅"
        fine_text = "No Fine (₹0) ✨"
        violation_flag = False
        conf_score = int(helmets[0]["conf"] * 100)
    else:
        # If model detected no helmet head vs helmet, default safe inspection
        helmet_status = "No Helmet Detected ❌"
        fine_text = "Fine: ₹1000 🚨"
        violation_flag = True
        conf_score = 75

    # Encode annotated image as base64 data URI for 100% reliable rendering without any proxy/404 issues
    import base64
    success, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if success:
        b64_str = base64.b64encode(buffer).decode('utf-8')
        annotated_html = f'<img src="data:image/jpeg;base64,{b64_str}" style="width: 100%; max-height: 500px; object-fit: contain; border-radius: 8px; border: 1px solid #E5E7EB;" alt="Annotated Output" />'
    else:
        annotated_html = "<p>Image processing error.</p>"

    # --- 4. Report Summary (Markdown) ---
    if violation_flag:
        report_md = f"""
### 🚨 Traffic Violation Challan Generated
- **Violation Category:** Riding Without Protective Helmet (Section 129, Motor Vehicles Act)
- **Compliance Status:** **No Helmet Detected** ({conf_score}% Model Confidence)
- **Vehicle Number Plate:** **`{final_plate_text}`**
- **Penalty Amount:** **₹1,000 INR**
- **Action Taken:** E-Challan Issued to Vehicle Owner
"""
    else:
        report_md = f"""
### ✅ Safe Driving - Traffic Rule Compliant
- **Compliance Status:** **Helmet Detected** ({conf_score}% Model Confidence)
- **Vehicle Number Plate:** **`{final_plate_text}`**
- **Penalty Amount:** **₹0 (No Fine)**
- **Action Taken:** Safe Rider Clearance Granted
"""

    return annotated_html, helmet_status, fine_text, final_plate_text, report_md