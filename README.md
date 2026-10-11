---
title: Smart Helmet Violation & Challan Detection
emoji: 🪖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
short_description: DL Project - Automatic Helmet & Number Plate Violation Detection
---

# 🪖 Smart Helmet Violation & Challan Detection System
> **Deep Learning Class Project**  
> Automated Motorcycle Safety Compliance & E-Challan Issuance

---

## 📌 Project Overview
This project provides an automated computer vision solution to monitor two-wheeler safety compliance:
1. **Helmet Detection:** Uses a **YOLOv8** deep convolutional neural network to detect whether riders are wearing protective helmets (`With Helmet`) or violating safety rules (`Without Helmet`).
2. **License Plate Localization & OCR:** Uses a secondary YOLOv8 model to detect vehicle number plates, and extracts alphanumeric characters using **EasyOCR** (CRAFT text detection + BiLSTM-CTC recognition).
3. **Automated Fine Generation:**
   - **Helmet Detected:** ✅ **No Fine (₹0)**
   - **No Helmet Detected:** ❌ **Fine: ₹1000** (under Section 129 of the Motor Vehicles Act).

---

## 🚀 How to Run Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Application:**
   ```bash
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://127.0.0.1:7860
   ```

---

## ☁️ How to Deploy to Hugging Face Spaces (in 3 Minutes)

1. **Sign Up / Log In:** Go to [Hugging Face](https://huggingface.co/).
2. **Create New Space:**
   - Click your profile icon at top right -> **New Space**.
   - **Space Name:** `helmet-violation-detection` (or your choice).
   - **License:** `MIT`.
   - **Select SDK:** Choose **Gradio**.
   - **Space Hardware:** Select **CPU basic** (Free).
   - Click **Create Space**.
3. **Upload Files:**
   - Go to the **Files** tab of your new Space.
   - Click **Add file** -> **Upload files**.
   - Upload the following files/folders from this project:
     - `app.py`
     - `logic.py`
     - `detect.py`
     - `requirements.txt`
     - `README.md`
     - `samples/` (contains sample test images)
     - `models/` (optional, models will auto-download from Hugging Face Hub if omitted)
   - Click **Commit changes to main**.
4. **Done!** Hugging Face will automatically build and start your application. You will have a live public URL to share for your DL class submission!

---

## 📊 Deep Learning Architecture

| Component | Model / Algorithm | Purpose |
| :--- | :--- | :--- |
| **Helmet Detection** | YOLOv8 Nano | High-speed detection of `With Helmet` vs `Without Helmet` |
| **Plate Detection** | YOLOv8 | Bounding box localization of vehicle license plate |
| **Character Recognition** | EasyOCR (CRAFT + BiLSTM) | Optical Character Recognition for registration plate text |
| **Decision Rule Engine**| Python Logic | Computes penalty fine (₹1000) and issues automated e-challan |

---

## 👥 Authors & Academic Submission
- **Project:** Deep Learning Laboratory / Course Project
- **Topic:** Object Detection and Automated Traffic Enforcement
