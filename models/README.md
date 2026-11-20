# ANPR AI Model Weights & Specifications

This directory contains the trained weights and neural network models powering the **AI-Based Automatic License Plate Detection and Recognition System (ANPR)**.

---

## 1. Model Summary Matrix

| Model File | Architecture / Backbone | Task | Dataset / Source | Parameters / Size |
|---|---|---|---|---|
| `yolov8n_license_plate.pt` | YOLOv8 Nano (CSPDarknet + C2f) | Dedicated License Plate Detection | Roboflow / Indian Vehicle Plate Dataset | 3.2M params (6.2 MB) |
| `yolov8n.pt` | YOLOv8 Nano (COCO Pretrained) | Multi-Class Vehicle Detection | MS-COCO (Car, Motorcycle, Bus, Truck) | 3.2M params (6.5 MB) |
| `easyocr/craft_mlt_25k.pth` | CRAFT (Character Region Awareness) | Character Bounding Region Localization | SynthText & MLT Multi-Language Text | 20.6M params (83.1 MB) |
| `easyocr/english_g2.pth` | ResNet + BiLSTM + CTC (CRNN) | Alphanumeric Character Sequence Decoding | Alphanumeric License Plate Corpus | 4.8M params (15.1 MB) |

---

## 2. Why a Dedicated License Plate Model is Required (Viva Question)

General object detectors (e.g., standard COCO YOLOv8) are trained to identify vehicles as holistic entities (`car`, `motorcycle`, `bus`, `truck`). They **do not possess a license plate bounding box class**.

Relying solely on vehicle detection forces an application to guess plate locations using arbitrary heuristics, leading to catastrophic failure under varying vehicle orientations and distances.

Our system incorporates a **dedicated fine-tuned YOLOv8 model** trained on thousands of vehicle license plates under various real-world conditions (daylight, nocturnal illumination, street glare, and skewed angles).

---

## 3. How to Obtain or Retrain the Model

If you wish to fine-tune the license plate detection model on custom regional datasets:

1. **Dataset**: Download an annotated dataset in YOLO format (e.g., [Roboflow Universe Vehicle License Plate Dataset](https://universe.roboflow.com/search?q=license+plate)).
2. **Training Command**:
   ```bash
   yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 batch=16
   ```
3. **Deploy**:
   Copy the resulting `runs/detect/train/weights/best.pt` to:
   `models/yolov8n_license_plate.pt`

---

## 4. Fallback Architecture

In environments where deep learning weights cannot be initialized or GPU acceleration is unavailable, `detection/plate_detector.py` automatically activates the **traditional OpenCV Contour Detector** (`detect_plates_opencv`), which uses:
- Grayscale conversion
- Bilateral filtering (11, 17, 17)
- Canny edge detection (30, 200)
- Morphological rectangular closing (13, 5)
- Aspect ratio contour filtering (1.8:1 to 6:1)

This dual-mode design demonstrates both deep learning and classical computer vision mastery during viva examinations.
