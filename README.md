---
title: Smart Helmet Violation & Challan Detection
emoji: 🪖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: false
license: mit
short_description: Helmet and Plate Detection DL Project
---

# 🪖 Smart Helmet Violation & Challan Detection System
> **Deep Learning Class Project**  
> Automated Motorcycle Safety Compliance, License Plate OCR & E-Challan Issuance

---

## 🌐 Live Application & Source Code
- 🔗 **Live Web App on Hugging Face:** **[https://huggingface.co/spaces/mahi-project/video](https://huggingface.co/spaces/mahi-project/video)**
- 🐙 **GitHub Repository:** **[https://github.com/mahioffical2005/helmet-violation-detection](https://github.com/mahioffical2005/helmet-violation-detection)**

---

## 📌 Project Overview
This project provides an automated, end-to-end computer vision system to monitor motorcycle and scooter traffic compliance in real time:

1. **Helmet Compliance Detection:**
   - Detects whether riders are wearing a helmet (`With Helmet`) or riding without headgear (`Without Helmet`) using a fine-tuned **YOLOv8 Nano** convolutional neural network.
2. **License Plate Localization & Intelligent OCR:**
   - Detects vehicle registration number plates using a dedicated YOLOv8 plate detector.
   - Extracts characters via **EasyOCR** (CRAFT text detector + BiLSTM-CTC recognition).
   - Features an **Indian License Plate Parser** that reconstructs multi-line registration numbers across all 36 state/UT codes into standard format (e.g. `TN 09 CF 7722`).
3. **Automated E-Challan & Penalty Engine:**
   - **Helmet Detected:** ✅ **No Fine (₹0)** — Safe driving clearance issued.
   - **No Helmet Detected:** ❌ **Fine: ₹1000** — Automated traffic violation challan generated under Section 129 of the Motor Vehicles Act.

---

## 📸 Real Test Sample Output

| Input Case | Detected Helmet Status | Fine Imposed | Extracted Vehicle Number Plate | Action |
| :--- | :--- | :--- | :--- | :--- |
| **Rider Without Helmet** (Sample 1) | `No Helmet Detected ❌` | **₹1,000 INR 🚨** | **`TN 09 CF 7722`** | 🚨 E-Challan Generated |
| **Rider Without Helmet** (Sample 2) | `No Helmet Detected ❌` | **₹1,000 INR 🚨** | **`AP 29 BK 7890`** | 🚨 E-Challan Generated |
| **Rider With Helmet** | `Helmet Detected ✅` | **₹0 (No Fine) ✨** | **`KA 05 MC 5678`** | ✅ Safe Clearance Granted |

---

## 📊 Deep Learning Architecture

| Component | Model / Algorithm | Purpose |
| :--- | :--- | :--- |
| **Helmet Detection Backbone** | YOLOv8 Nano (CNN) | High-speed real-time classification of `With Helmet` vs `Without Helmet` |
| **License Plate Detector** | YOLOv8 Object Detection | Bounding box localization of vehicle license plate region |
| **Text Recognition (OCR)** | EasyOCR (CRAFT + ResNet-BiLSTM) | Character extraction from localized number plate crop |
| **Plate Format Parser** | Regex & Spatial Reordering | Aligns two-line Indian registration plates (e.g. `TN 09 CF 7722`) |
| **Rule & Challan Engine** | Python Backend | Applies Section 129 Motor Vehicles Act fine computation (₹1000) |

---

## 🚀 How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/mahioffical2005/helmet-violation-detection.git
   cd helmet-violation-detection
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Application:**
   ```bash
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://127.0.0.1:7860
   ```

---

## 📁 Project Structure

```
├── app.py              # Gradio Web Application UI
├── logic.py            # Deep Learning detection pipeline, OCR & Challan logic
├── detect.py           # Helper script for standalone YOLO inference
├── requirements.txt    # Python dependencies for local & cloud deployment
├── README.md           # Project documentation and Hugging Face metadata
├── models/             # YOLOv8 deep learning model weights
└── samples/            # Test images for demonstration
```

---

## 👥 Authors & Academic Submission
- **Project:** Deep Learning Laboratory / Course Project
- **Topic:** Object Detection, License Plate OCR & Automated Traffic Enforcement
- **Tech Stack:** PyTorch, Ultralytics YOLOv8, EasyOCR, Gradio, Python
