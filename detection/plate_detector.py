"""
detection/plate_detector.py
License Plate Detection Module supporting:
1. Deep Learning YOLOv8 Model (Dedicated trained weights for license plates)
2. Traditional OpenCV Contour & Morphological Detector (Fallback / Viva demonstration)
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO

DEFAULT_PLATE_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "yolov8n_license_plate.pt"
)


class PlateDetector:
    def __init__(self, model_path: str = DEFAULT_PLATE_WEIGHTS):
        self.model_path = model_path
        self.yolo_model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self):
        """Loads the dedicated YOLOv8 license plate detection weights."""
        if os.path.exists(self.model_path):
            try:
                self.yolo_model = YOLO(self.model_path)
                print(f"[PlateDetector] Dedicated license plate YOLO weights loaded from {self.model_path}")
            except Exception as e:
                print(f"[PlateDetector] Error loading YOLO plate model: {e}")
                self.yolo_model = None
        else:
            print(f"[PlateDetector] Dedicated weights not found at {self.model_path}. OpenCV mode will serve as fallback.")
            self.yolo_model = None

    def detect_plates_yolo(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Detects license plates using YOLOv8 neural network.
        Returns list of detections with bbox, confidence, and method.
        """
        if self.yolo_model is None or image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        detections: List[Dict[str, Any]] = []

        try:
            results = self.yolo_model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1 = max(0, min(int(xyxy[0]), w - 1))
                    y1 = max(0, min(int(xyxy[1]), h - 1))
                    x2 = max(x1 + 10, min(int(xyxy[2]), w))
                    y2 = max(y1 + 10, min(int(xyxy[3]), h))

                    # Crop plate
                    crop = image[y1:y2, x1:x2].copy()

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": round(conf, 3),
                        "method": "YOLOv8 Plate Detector",
                        "crop": crop
                    })
        except Exception as e:
            print(f"[PlateDetector] YOLO inference error: {e}")

        return detections

    def detect_plates_opencv(
        self,
        image: np.ndarray,
        min_aspect_ratio: float = 1.6,
        max_aspect_ratio: float = 6.2,
        min_area: int = 800,
        max_area: int = 150000
    ) -> List[Dict[str, Any]]:
        """
        Traditional Computer Vision License Plate Detector:
        - Grayscale conversion
        - Bilateral Filtering (edge-preserving smoothing)
        - Canny Edge Detection
        - Morphological Closing (connect character contours into a plate blob)
        - Contour Analysis and aspect ratio filtering
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Bilateral filter for noise reduction while keeping edges crisp
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

        # Canny edge detector
        edged = cv2.Canny(bfilter, 30, 200)

        # Morphological Closing with horizontal rectangular kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # Find external contours
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        detections: List[Dict[str, Any]] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue

            # Approximate contour to polygon
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / float(bh) if bh > 0 else 0

            # Indian plates typically have aspect ratio between 2.0 and 5.5
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                # Add small padding
                pad_x = int(bw * 0.05)
                pad_y = int(bh * 0.05)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2 = min(h, y + bh + pad_y)

                crop = image[y1:y2, x1:x2].copy()
                confidence = 0.70 if len(approx) == 4 else 0.55

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "method": "OpenCV Morphological",
                    "crop": crop
                })

                if len(detections) >= 3:
                    break

        return detections

    def detect_plates(
        self,
        image: np.ndarray,
        method: str = "auto",
        conf_threshold: float = 0.30
    ) -> List[Dict[str, Any]]:
        """
        Unified detection entry point.
        Methods: 'auto', 'yolo', 'opencv'
        """
        if method == "opencv":
            return self.detect_plates_opencv(image)

        # Auto or YOLO
        detections = []
        if self.yolo_model is not None:
            detections = self.detect_plates_yolo(image, conf_threshold=conf_threshold)

        # If YOLO found nothing or model unavailable, fallback to OpenCV
        if not detections and (method == "auto" or self.yolo_model is None):
            detections = self.detect_plates_opencv(image)

        return detections
