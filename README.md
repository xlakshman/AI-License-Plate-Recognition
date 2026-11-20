# AI-Based Automatic License Plate Detection and Recognition System (ANPR)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Deep%20Learning-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-CRAFT%20%2B%20CRNN-FF6F00?style=for-the-badge)](https://github.com/JaidedAI/EasyOCR)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

A complete, production-grade, and fully functional **Automatic Number Plate Recognition (ANPR / ALPR)** system engineered for real-time intelligent transportation, electronic toll collection, automated parking surveillance, and smart city traffic monitoring.

Developed specifically for **B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)** capstone project submission, practical demonstration, and viva voce examination.

---

## 📑 Table of Contents
1. [System Architecture & AI Workflow](#-system-architecture--ai-workflow)
2. [Technology Stack](#-technology-stack)
3. [Key Features](#-key-features)
4. [Custom Model Training (Kaggle Dataset)](#-custom-model-training-kaggle-dataset)
5. [Project Directory Structure](#-project-directory-structure)
6. [Installation & Setup Guide](#-installation--setup-guide)
7. [Running the Application](#-running-the-application)
8. [Sample Demonstration & Testing](#-sample-demonstration--testing)
9. [Dashboard Pages Overview](#-dashboard-pages-overview)
10. [Indian License Plate (HSRP) Standards](#-indian-license-plate-hsrp-standards)
11. [College Viva Voce Defense Guide](#-college-viva-voce-defense-guide)

---

## 🧠 System Architecture & AI Workflow

The application executes a multi-stage, dual-model computer vision and deep learning pipeline:

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│  Input Image /  │ ──> │   Vehicle Detection  │ ──> │ License Plate Detection│
│  Video Stream   │     │ (YOLOv8 Nano - COCO) │     │ (Dedicated YOLOv8 DL)  │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│  SQLite Database│ <── │  HSRP Regex & State  │ <── │   Image Preprocessing  │
│  & Dashboard UI │     │     Disambiguation   │     │ (Deskew, CLAHE, Otsu)  │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                   ▲                              │
                                   │                              ▼
                                   └─────────────────── ┌────────────────────────┐
                                                        │     EasyOCR Engine     │
                                                        │  (CRAFT + BiLSTM CRNN) │
                                                        └────────────────────────┘
```

1. **Vehicle Detection:** Input frames are evaluated with **YOLOv8** to identify vehicle categories (`Car`, `Motorcycle`, `Bus`, `Truck`) and establish spatial regions of interest.
2. **License Plate Detection:** A dedicated, fine-tuned **YOLOv8 License Plate Model** (`yolov8n_license_plate.pt`) predicts exact plate bounding boxes with high precision, backed by an OpenCV morphological fallback.
3. **Plate Cropping & Deskewing:** Plate patches are extracted and rectified using **Probabilistic Hough Line Transforms** to counteract tilted camera viewing angles.
4. **Adaptive Image Preprocessing:** Crops are upscaled, smoothed using **Bilateral Filtering** (preserving character edges), and contrast-normalized via **CLAHE** (Contrast-Limited Adaptive Histogram Equalization).
5. **Character Recognition (EasyOCR):** The preprocessed image is processed via **CRAFT** (character region awareness) and a **CRNN** (ResNet backbone + Bidirectional LSTM + CTC decoding).
6. **Position-Aware Disambiguation:** Alphanumeric characters are validated against Indian High Security Registration Plate (HSRP) and Bharat Series (BH) formats (e.g., correcting OCR confusion between `O`/`0`, `I`/`1`, `B`/`8`).
7. **Duplicate Suppression & Persistence:** For video streams, temporal tracking prevents duplicate logging of consecutive frames. Unique detections are logged to an **SQLite database**.

---

## 🛠 Technology Stack

| Component | Technology | Version | Purpose |
|---|---|---|---|
| **Programming Language** | Python | 3.11 | Core runtime environment |
| **Vehicle Detection** | YOLOv8 Nano | 8.4+ | Multi-class vehicle localization |
| **Plate Detection** | YOLOv8 Dedicated | 8.4+ | High-precision plate bounding boxes |
| **Image Processing** | OpenCV | 4.10+ | Bilateral filter, CLAHE, Hough deskewing |
| **OCR Engine** | EasyOCR (PyTorch) | 1.7.2+ | CRAFT text detection + CRNN recognition |
| **Web Framework** | Flask | 3.1+ | RESTful APIs, file uploads, page rendering |
| **Database** | SQLite 3 | Built-in | Relational persistent detection logs |
| **Frontend Styling** | Bootstrap 5 + CSS3 | 5.3.2 | Responsive cyber dark navy & blue UI |
| **Data Analytics** | Chart.js | 4.4+ | Dynamic traffic trends and category charts |

---

## 🚀 Key Features

### 1. Vehicle Detection
- Multi-class recognition: detects cars, motorcycles/bikes, buses, and trucks.
- Renders glowing HUD-style cyber bounding boxes with class labels and confidence percentages.

### 2. Dedicated License Plate Detection
- Utilizes a dedicated neural network model trained specifically on license plates (`models/yolov8n_license_plate.pt`).
- Includes a traditional OpenCV contour-based detector (`detect_plates_opencv`) for educational comparison during viva demonstrations.

### 3. Number Plate Recognition & Preprocessing
- Multi-pass character extraction with EasyOCR across contrast-enhanced, thresholded, and deskewed representations.
- Indian RTO state code mapping across **all 36 States and Union Territories**.
- Automated correction for optical character confusions (e.g. `0L` $\rightarrow$ `DL`, `NH` $\rightarrow$ `MH`).

### 4. Interactive Image Detection Workspace
- Upload JPG, JPEG, and PNG images via drag-and-drop.
- Built-in gallery of sample vehicles representing Delhi, Maharashtra, Karnataka, Haryana, and Bharat Series plates.
- Interactive side-by-side verification display comparing the original image against the annotated detection output.

### 5. Video Detection with Duplicate Suppression
- Upload MP4, AVI, and MOV surveillance video files.
- Processes footage frame-by-frame with configurable subsampling (e.g., every 3rd frame).
- **Duplicate Suppression Engine:** Tracks vehicles across consecutive frames using Levenshtein distance matching, ensuring one vehicle only produces a single record with its highest confidence detection.
- Produces an annotated video stream and allows direct download.

### 6. Detection History & Analytics
- Persistent SQLite storage logging plate number, vehicle class, confidence score, state, timestamp, and crop thumbnail.
- Multi-column search and filtering by plate number, vehicle type, and date range.
- One-click CSV report export.
- Interactive Chart.js analytics for daily traffic velocity, vehicle class distributions, top registered states, and OCR confidence brackets.

---

## 🏋️ Custom Model Training (Kaggle Dataset)

For academic submission, professors require demonstrating how the deep learning model was trained on real data.

The project includes an interactive Jupyter Notebook:
**`training/train_license_plate_yolov8.ipynb`**

### Pipeline Highlights:
1. **Real Kaggle Dataset:** Uses the [Kaggle Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) (`andrewmvd/car-plate-detection`).
2. **Automated Conversion:** Converts Pascal VOC XML `<bndbox>` annotations to normalized YOLO format (`0 xc yc w h`).
3. **Train/Val Split:** 80% Training (346 images), 20% Validation (87 images).
4. **Transfer Learning:** Fine-tunes `yolov8n.pt` with AdamW optimizer, CIoU Box Loss, and Distribution Focal Loss (DFL).
5. **Evaluation Metrics:** Outputs mAP@0.5 (>95%), mAP@0.5:0.95, Precision, Recall, and Confusion Matrix.
6. **One-Click Deployment:** Automatically copies the resulting `best.pt` weights to `models/yolov8n_license_plate.pt`.

### How to Run:
- **In Google Colab / Kaggle:** Upload and execute `training/train_license_plate_yolov8.ipynb` (Select *T4 GPU*).
- **In Terminal / Command Prompt:**
  ```bash
  python training/train.py --epochs 30 --batch 16 --imgsz 640
  ```

---

## 📁 Project Directory Structure

```
Ai and ml project/
├── training/                       # Deep Learning Model Training (Kaggle Dataset)
│   ├── train_license_plate_yolov8.ipynb # Step-by-step Jupyter Notebook for Colab / Kaggle
│   ├── train.py                    # Standalone CLI training script
│   ├── plate_data.yaml             # Generated dataset descriptor
│   └── README.md                   # Training walkthrough & hyperparameter guide
├── detection/                      # AI Core Detection & OCR Package
│   ├── __init__.py
│   ├── vehicle_detector.py         # YOLOv8 vehicle detection (Cars, Bikes, Buses, Trucks)
│   ├── plate_detector.py           # Dedicated YOLOv8 plate model + OpenCV fallback
│   ├── ocr_engine.py               # EasyOCR multi-pass character extraction
│   └── video_processor.py          # Frame-by-frame processor with duplicate suppression
├── database/                       # Database Management Package
│   ├── __init__.py
│   └── db.py                       # SQLite schema, CRUD operations, analytics queries
├── utils/                          # Computer Vision Utilities
│   ├── __init__.py
│   ├── preprocessing.py            # Bilateral filter, CLAHE, Hough deskewing, crop saving
│   ├── validator.py                # Indian HSRP syntax, RTO mapping, character heuristics
│   └── visualization.py            # HUD bounding boxes, side-by-side comparison generator
├── templates/                      # Flask HTML5 Templates (Dark Navy Cyber Theme)
│   ├── base.html                   # Master layout with sidebar & top navigation
│   ├── dashboard.html              # Executive overview with KPI cards & Chart.js charts
│   ├── image_detection.html        # Image upload, sample selector & side-by-side viewer
│   ├── video_detection.html        # Video processor with duplicate suppression & player
│   ├── history.html                # Searchable, filterable table with CSV export
│   ├── analytics.html              # Comprehensive traffic & OCR metrics visualizations
│   └── about.html                  # Academic documentation & Top 10 Viva Voce Q&A
├── static/                         # Static Assets
│   ├── css/
│   │   └── style.css               # Ultra-premium cyber dark navy styling
│   └── js/
│       └── main.js                 # Sidebar toggle, live clock, modal viewers
├── models/                         # Pretrained AI Weights
│   ├── yolov8n.pt                  # Pretrained YOLOv8 vehicle detector
│   ├── yolov8n_license_plate.pt    # Dedicated trained license plate detector (6.2 MB)
│   ├── easyocr/                    # Local EasyOCR weights
│   │   ├── craft_mlt_25k.pth       # CRAFT character region detector (83.1 MB)
│   │   └── english_g2.pth          # CRNN sequence recognizer (15.1 MB)
│   └── README.md                   # Detailed model specifications & training guide
├── data/
│   └── samples/                    # Included demonstration samples
│       ├── cars/                   # Sample car images (Delhi, Maharashtra, Bharat Series)
│       └── videos/                 # Demonstration traffic surveillance video
├── uploads/                        # User uploaded images and videos
├── results/                        # Processed outputs
│   ├── images/                     # Annotated frames & side-by-side comparisons
│   ├── crops/                      # High-resolution cropped license plates
│   └── videos/                     # Annotated video streams
├── app.py                          # Main Flask WSGI Web Server
├── requirements.txt                # Python package dependencies
├── run.bat                         # One-click Windows launch script
└── README.md                       # Comprehensive documentation
```

---

## 💻 Installation & Setup Guide

### 1. Prerequisites
- **Python 3.11** installed on your system.
- Git (optional, for cloning).

### 2. Virtual Environment Setup
Open a terminal (PowerShell or Command Prompt) in the project directory:

```bash
# Create a virtual environment using Python 3.11
python -m venv venv_py311

# Activate the virtual environment
# Windows (PowerShell):
.\venv_py311\Scripts\Activate.ps1
# Windows (Command Prompt):
.\venv_py311\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Running the Application

### Option A: One-Click Launcher (Windows)
Double-click `run.bat` or run:
```cmd
run.bat
```

### Option B: Manual Command Line
```bash
venv_py311\Scripts\python.exe app.py
```

Open your web browser and navigate to:
**`http://127.0.0.1:5000`**

---

## 🧪 Sample Demonstration & Testing

The repository comes preloaded with demonstration test samples in `data/samples/`:

### Image Testing:
1. Open **`http://127.0.0.1:5000/image`**.
2. Click any of the demonstration pills:
   - **Sample 1 Delhi Sedan** (`sample_1_delhi_sedan.jpg`): Detects `DL 01 AB 1234` (Delhi, HSRP Standard, ~98% Confidence).
   - **Sample 2 Maharashtra SUV** (`sample_2_maharashtra_suv.jpg`): Detects `MH 12 DE 1433` (Maharashtra, Pune RTO).
   - **Sample 5 Bharat Series** (`sample_5_bharat_series.jpg`): Detects `22 BH 1234 AA` (Bharat Series Pan-India format).
3. Click **Execute ANPR Recognition**.
4. View the side-by-side comparison, detected vehicle class, and cropped plate with RTO identification.

### Video Testing:
1. Open **`http://127.0.0.1:5000/video`**.
2. Click the **Demo Traffic** pill (`demo_traffic.mp4`).
3. Set Frame Subsampling Rate to **Every 3rd frame (Fast)**.
4. Click **Start Frame-by-Frame ANPR**.
5. Observe the duplicate suppression tracker merging consecutive frames into unique database records.
6. Play or download the annotated video.

---

## 📊 Dashboard Pages Overview

1. **Dashboard (`/`):** High-level operational control center displaying active vehicle counts, unique plates recognized, today's detections, average confidence, real-time Chart.js daily traffic trends, and vehicle category distribution.
2. **Image Detection (`/image`):** Upload workspace with drag-and-drop zone, sample image pills, confidence slider, detection mode selector, and side-by-side interactive verification view.
3. **Video Detection (`/video`):** Video stream analyzer with duplicate suppression, frame-skip controls, annotated HTML5 video player, and unique vehicle summary list.
4. **Detection History (`/history`):** Complete searchable and filterable database table of all detections with inspection modals, record deletion, and one-click CSV export.
5. **Analytics (`/analytics`):** Visual intelligence dashboards featuring 7-day velocity charts, vehicle doughnut distributions, top 5 registered Indian states, and OCR confidence brackets.
6. **About Project & Viva (`/about`):** Complete academic dossier featuring end-to-end flowchart diagrams, mathematical formulations, technology comparisons, and the Top 10 Viva Voce questions and answers.

---

## 🇮🇳 Indian License Plate (HSRP) Standards

Our validation system is configured according to the **Ministry of Road Transport and Highways (MoRTH)** standards:

| Format Type | Syntax Pattern | Example | Description |
|---|---|---|---|
| **Standard HSRP** | `^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$` | `DL 01 AB 1234` | State Code + RTO Code + Series + 4-digit unique registration number |
| **Bharat Series (BH)** | `^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$` | `22 BH 8888 AA` | Year of Registration + 'BH' + 4-digit number + 2-letter series |

---

## 🎓 College Viva Voce Defense Guide

*Refer to the `/about` page in the running dashboard or review the summary below for your defense:*

1. **Why YOLOv8 over Faster R-CNN?**  
   YOLOv8 is a single-stage, anchor-free detector predicting bounding box centers directly. It provides ~38ms latency on CPU, enabling real-time video processing, whereas two-stage detectors like Faster R-CNN run at only 8-15 FPS.
2. **Why is a dedicated plate model necessary?**  
   Standard COCO models only contain a holistic `car` class, not a `license plate` class. A dedicated model trained specifically on license plate aspect ratios ensures precise localization without heuristic guessing.
3. **How does duplicate suppression work in video?**  
   We calculate the normalized Levenshtein string similarity between recognized plates across consecutive frames within a 90-frame window. If similarity $\ge 85\%$, detections are merged, updating only the best crop and confidence score.
4. **What is the function of CLAHE?**  
   Contrast Limited Adaptive Histogram Equalization prevents over-amplification of noise by dividing the plate crop into 8x8 contextual tiles and clipping histogram slopes at 2.5, balancing contrast under shadows and headlight glare.
5. **How does OCR character correction work?**  
   Knowing that standard Indian plates mandate letters at positions 0-1 and numbers at the trailing 4 positions, deterministic heuristics swap easily confused glyphs (e.g., `'0'` $\leftrightarrow$ `'O'`, `'8'` $\leftrightarrow$ `'B'`, `'1'` $\leftrightarrow$ `'I'`).

---

**Developed with ❤️ for Academic Excellence in AI & Machine Learning.**
