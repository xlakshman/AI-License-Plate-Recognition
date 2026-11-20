"""
training/train.py
Standalone Deep Learning Training Script for YOLOv8 License Plate Detection on Real Kaggle Dataset.
Features automatic dataset acquisition, Pascal VOC to YOLO annotation conversion,
train/val splitting, transfer learning, validation evaluation, and deployment to models/.
"""

import os
import sys
import shutil
import random
import argparse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

# Project paths
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_DIR = TRAINING_DIR / "kaggle_raw_dataset"
YOLO_DATA_DIR = TRAINING_DIR / "dataset_yolo"


def check_environment():
    """Checks Python, PyTorch, and CUDA device status."""
    print("=" * 65)
    print("  YOLOv8 LICENSE PLATE DETECTION MODEL TRAINING")
    print("  Academic Project: B.Tech CSE (AI & ML)")
    print("=" * 65)
    print(f"[*] Python Version  : {sys.version.split()[0]}")
    print(f"[*] PyTorch Version : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"[*] CUDA Available  : {cuda_avail}")
    if cuda_avail:
        print(f"[*] GPU Name        : {torch.cuda.get_device_name(0)}")
        print(f"[*] Total VRAM      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        return 0
    else:
        print("[*] Running on multi-core CPU mode.")
        return "cpu"


def download_kaggle_dataset():
    """Downloads real-world car license plate dataset from Kaggle or direct mirror."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = RAW_DATA_DIR / "images"
    annotations_dir = RAW_DATA_DIR / "annotations"

    if images_dir.exists() and annotations_dir.exists() and len(list(images_dir.glob("*"))) > 50:
        print(f"[+] Dataset already present in {RAW_DATA_DIR.resolve()}")
        return images_dir, annotations_dir

    print("[*] Downloading Kaggle Car Plate Detection Dataset...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("andrewmvd/car-plate-detection")
        print(f"[+] Kagglehub downloaded to: {path}")
        return Path(path) / "images", Path(path) / "annotations"
    except Exception as e:
        print(f"[*] Note on kagglehub ({e}), using direct dataset mirror...")

    zip_file = RAW_DATA_DIR / "car-plate-detection.zip"
    if not zip_file.exists():
        mirror_urls = [
            "https://raw.githubusercontent.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide/main/car-plate-detection.zip",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/car-plate-detection.zip"
        ]
        downloaded = False
        for url in mirror_urls:
            try:
                print(f"[*] Downloading dataset archive from: {url}")
                urllib.request.urlretrieve(url, zip_file)
                downloaded = True
                break
            except Exception as dl_err:
                print(f"[*] Mirror failed ({dl_err}), trying next...")
        if not downloaded:
            raise RuntimeError("Could not download dataset from any mirror. Check internet connection.")

    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(RAW_DATA_DIR)

    return RAW_DATA_DIR / "images", RAW_DATA_DIR / "annotations"


def voc_xml_to_yolo(xml_path: Path):
    """Converts a Pascal VOC XML annotation to YOLO normalized format [class, xc, yc, w, h]."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        # Mathematical normalization:
        # xc = (xmin + xmax) / (2 * width)
        # yc = (ymin + ymax) / (2 * height)
        # w  = (xmax - xmin) / width
        # h  = (ymax - ymin) / height
        xc = ((xmin + xmax) / 2.0) / width
        yc = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height

        boxes.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return boxes


def prepare_yolo_dataset(raw_images_dir: Path, raw_ann_dir: Path):
    """Structures images and labels into train/val 80/20 split with data.yaml."""
    for split in ["train", "val"]:
        (YOLO_DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_list = sorted([f for f in raw_images_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    pairs = []
    for img_p in img_list:
        xml_p = raw_ann_dir / f"{img_p.stem}.xml"
        if xml_p.exists():
            pairs.append((img_p, xml_p))

    print(f"[+] Total verified image-annotation pairs: {len(pairs)}")
    random.seed(42)
    random.shuffle(pairs)

    split_idx = int(0.80 * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"[+] Split: {len(train_pairs)} Train images (80%), {len(val_pairs)} Val images (20%)")

    def process_split(split_pairs, split_name):
        for img_p, xml_p in split_pairs:
            # Copy image
            shutil.copy(img_p, YOLO_DATA_DIR / "images" / split_name / img_p.name)
            # Convert XML to YOLO txt
            lines = voc_xml_to_yolo(xml_p)
            label_p = YOLO_DATA_DIR / "labels" / split_name / f"{img_p.stem}.txt"
            with open(label_p, "w") as f:
                f.write("\n".join(lines) + "\n")

    process_split(train_pairs, "train")
    process_split(val_pairs, "val")

    # Generate data.yaml
    yaml_config = {
        "path": str(YOLO_DATA_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["license_plate"]
    }
    yaml_file = TRAINING_DIR / "plate_data.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    print(f"[+] Configured dataset descriptor: {yaml_file}")
    return yaml_file


def train_model(args, device):
    """Executes YOLOv8 model training and quantitative validation."""
    print("\n" + "=" * 65)
    print("  INITIALIZING YOLOv8 NANO BACKBONE FOR TRANSFER LEARNING")
    print("=" * 65)
    
    # 1. Download dataset and prepare
    raw_imgs, raw_anns = download_kaggle_dataset()
    yaml_file = prepare_yolo_dataset(raw_imgs, raw_anns)

    # 2. Load model
    base_weights = args.weights
    print(f"[*] Loading base model weights from: {base_weights}")
    model = YOLO(base_weights)

    # 3. Train
    print("\n[*] Commencing supervised fine-tuning...")
    print(f"[*] Hyperparameters: Epochs={args.epochs}, Batch={args.batch}, ImgSz={args.imgsz}, Optimizer={args.optimizer}")

    results = model.train(
        data=str(yaml_file),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr,
        patience=args.patience,
        device=device,
        augment=True,
        project=str(TRAINING_DIR / "runs"),
        name="plate_yolov8_model",
        exist_ok=True,
        verbose=True
    )

    print("\n" + "=" * 65)
    print("  QUANTITATIVE MODEL VALIDATION")
    print("=" * 65)
    val_res = model.val()
    metrics = val_res.results_dict
    mAP50 = metrics.get("metrics/mAP50(B)", 0)
    mAP50_95 = metrics.get("metrics/mAP50-95(B)", 0)
    precision = metrics.get("metrics/precision(B)", 0)
    recall = metrics.get("metrics/recall(B)", 0)

    print(f"[+] Mean Average Precision (mAP@0.5)     : {mAP50:.4f} ({mAP50*100:.1f}%)")
    print(f"[+] Mean Average Precision (mAP@0.5:0.95): {mAP50_95:.4f} ({mAP50_95*100:.1f}%)")
    print(f"[+] Precision (P)                        : {precision:.4f} ({precision*100:.1f}%)")
    print(f"[+] Recall (R)                           : {recall:.4f} ({recall*100:.1f}%)")

    # 4. Deploy weights
    best_weights = TRAINING_DIR / "runs" / "plate_yolov8_model" / "weights" / "best.pt"
    if best_weights.exists():
        deploy_path = MODELS_DIR / "yolov8n_license_plate.pt"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights, deploy_path)
        print("\n" + "=" * 65)
        print("  MODEL DEPLOYMENT COMPLETE")
        print("=" * 65)
        print(f"[+] Successfully deployed best weights to: {deploy_path.resolve()}")
        print(f"[+] Model Size: {deploy_path.stat().st_size / (1024**2):.2f} MB")
        print("[+] The Flask ANPR application is now powered by your custom trained model!")


def main():
    parser = argparse.ArgumentParser(description="Train custom YOLOv8 model for License Plate Detection on Kaggle dataset.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution (default: 640)")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer (default: AdamW)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pretrained weights backbone")
    args = parser.parse_args()

    device = check_environment()
    train_model(args, device)


if __name__ == "__main__":
    main()
