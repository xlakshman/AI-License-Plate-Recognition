# YOLOv8 Custom License Plate Detection Model Training

This directory contains the complete deep learning training pipeline for training a dedicated **YOLOv8** model on a **real-world Kaggle dataset** for vehicle license plate detection.

---

## 📁 Directory Files

| File | Type | Description |
|---|---|---|
| `train_license_plate_yolov8.ipynb` | Jupyter Notebook | Interactive, step-by-step training notebook with visual outputs, loss curves, and ground-truth inspection. Ideal for Google Colab, Kaggle, or VS Code. |
| `train.py` | Python Script | Professional command-line training script with argparse hyperparameters. |
| `plate_data.yaml` | YAML Config | Generated YOLOv8 dataset configuration file. |
| `README.md` | Documentation | Complete training, hyperparameter, and viva voce defense guide. |

---

## 📊 Dataset Specification

- **Source:** [Kaggle Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) (`andrewmvd/car-plate-detection`)
- **Total Images:** 433 real road vehicle images across varying angles, weather, and illumination conditions.
- **Annotation Format:** Pascal VOC XML (`<bndbox>` with `xmin, ymin, xmax, ymax`).
- **Target Format:** YOLOv8 normalized format (`0 x_center y_center width height`).
- **Split Ratio:** 80% Training (346 images), 20% Validation (87 images).

---

## 🚀 How to Run Model Training

### Option 1: In Google Colab or Kaggle (Recommended for Free GPU)

1. Upload `training/train_license_plate_yolov8.ipynb` to Google Colab or Kaggle Notebooks.
2. In Google Colab, enable GPU:
   - Go to **Runtime > Change runtime type > T4 GPU**.
3. Run the cells step-by-step:
   - **Step 1-2:** Environment check and dependency installation.
   - **Step 3:** Automated dataset download from Kaggle.
   - **Step 4-5:** VOC XML to YOLO format conversion and 80/20 train/val splitting.
   - **Step 6:** Visual sanity check plotting ground-truth bounding boxes.
   - **Step 7-9:** YOLOv8 Nano transfer learning and model training (30 epochs).
   - **Step 10-11:** Quantitative validation (mAP@0.5, mAP@0.5:0.95, Precision, Recall).
   - **Step 12:** Real-world inference testing and plate cropping.
   - **Step 13:** Export to ONNX and deployment to `models/yolov8n_license_plate.pt`.

---

### Option 2: Running Locally via Python CLI

Open your terminal in the project directory:

```bash
# 1. Activate your virtual environment
.\venv_py311\Scripts\Activate.ps1

# 2. Run the training script with default parameters (30 epochs, batch 16, 640x640)
python training/train.py

# Optional: Customize hyperparameters
python training/train.py --epochs 50 --batch 8 --imgsz 640 --lr 0.001 --optimizer AdamW
```

---

## 📐 Mathematical Formulation of Bounding Box Normalization

During training data preparation, pixel coordinates from Pascal VOC are converted to normalized coordinates for YOLO:

$$x_{\text{center}} = \frac{x_{\min} + x_{\max}}{2 \cdot W}$$

$$y_{\text{center}} = \frac{y_{\min} + y_{\max}}{2 \cdot H}$$

$$w = \frac{x_{\max} - x_{\min}}{W}$$

$$h = \frac{y_{\max} - y_{\min}}{H}$$

Where:
- $W$ is the image width in pixels.
- $H$ is the image height in pixels.
- Coordinates $x_{\text{center}}, y_{\text{center}}, w, h$ are normalized within $[0.0, 1.0]$.

---

## 🎯 Model Evaluation Metrics & Loss Functions

### 1. Complete IoU Loss (CIoU)
Measures overlap between predicted bounding box $B$ and ground truth $B^{gt}$, while penalizing distance between their center points and aspect ratio discrepancy:
$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

### 2. Distribution Focal Loss (DFL)
Optimizes bounding box boundaries as continuous probability distributions rather than single rigid values, improving localization on blurry or partial plates.

### 3. Mean Average Precision (mAP)
- **mAP@0.5:** Precision at an Intersection-over-Union (IoU) threshold of 0.50 (typically $>94\%$ on this dataset).
- **mAP@0.5:0.95:** Average mAP across 10 IoU thresholds from 0.50 to 0.95 with step 0.05 (the primary COCO benchmark metric).

---

## 🎓 College Viva Voce Q&A on Model Training

### Q1: Why did you choose transfer learning from `yolov8n.pt` rather than training from scratch?
> **Answer:** Training a deep neural network from scratch requires hundreds of thousands of images and days of compute to learn basic spatial features (edges, corners, textures). By using transfer learning with pre-trained weights from the COCO dataset (`yolov8n.pt`), the model already possesses rich feature extraction filters in its CSPDarknet backbone. We only fine-tune the detection head and higher convolutional layers on the license plate dataset, which converges in 30 epochs with high accuracy.

### Q2: What is the purpose of Data Augmentation during training?
> **Answer:** We enable Mosaic augmentation, random horizontal flipping, scaling, and HSV color space jitter. Mosaic augmentation stitches 4 training images together into one frame, forcing the model to learn to identify smaller plates and vehicles across diverse contextual surroundings.

### Q3: How do you choose between `yolov8n` (nano) and `yolov8s` / `yolov8m`?
> **Answer:** In automated license plate recognition, real-time inference speed is essential for high-FPS video surveillance. `yolov8n` has only 3.2 million parameters and requires ~8.7 GFLOPs, delivering 35–45 FPS on standard CPUs and >120 FPS on GPUs. Larger models like `yolov8m` provide only marginal accuracy gains on rectangular plates while doubling the computational latency.
