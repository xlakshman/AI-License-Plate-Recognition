"""
src/detector.py
License Plate Detector supporting:
1. YOLOv8 Deep Learning Detector (Pretrained / Fine-tuned)
2. Traditional OpenCV Contour / Morphological Detector (Fallback / Educational Viva mode)
3. Full Educational Pipeline Inspection for viva demonstrations
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ultralytics import YOLO

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolov8n_license_plate.pt")


class PlateDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.yolo_model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self):
        """Loads the YOLOv8 model if the weights file exists."""
        if os.path.exists(self.model_path):
            try:
                self.yolo_model = YOLO(self.model_path)
            except Exception as e:
                print(f"[Detector] Warning: Could not load YOLO weights from {self.model_path}: {e}")
                self.yolo_model = None
        else:
            print(f"[Detector] Model file not found at {self.model_path}. Traditional CV detector active.")

    def detect_plates_yolo(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Detects license plates using YOLOv8 neural network.
        Returns list of detections with [x1, y1, x2, y2], confidence, and method='YOLOv8'.
        """
        if self.yolo_model is None:
            return []

        results = self.yolo_model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        detections = []
        h, w = image.shape[:2]

        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                # Clamp coordinates inside image boundary
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 10, min(x2, w))
                y2 = max(y1 + 5, min(y2, h))

                confidence = float(box.conf[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "method": "YOLOv8 Deep Learning"
                })

        return detections

    def detect_plates_opencv(
        self,
        image: np.ndarray,
        min_aspect_ratio: float = 1.8,
        max_aspect_ratio: float = 6.0,
        min_area: int = 1200,
        max_area: int = 150000
    ) -> List[Dict[str, Any]]:
        """
        Traditional Computer Vision License Plate Detector:
        - Grayscale conversion
        - Bilateral Filtering (removes noise while preserving edges)
        - Canny Edge Detection
        - Morphological Rectangular Closing
        - Contour Analysis: Filter contours matching vehicle plate aspect ratio (approx 2:1 to 5:1)
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Bilateral filter for edge-preserving noise reduction
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

        # Canny Edge detection
        edged = cv2.Canny(bfilter, 30, 200)

        # Morphological Closing using horizontal kernel to connect character edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:25]

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_area <= area <= max_area):
                continue

            x, y, cw, ch = cv2.boundingRect(c)
            if ch == 0:
                continue

            aspect_ratio = float(cw) / ch

            # Check aspect ratio typical of plates
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                # Approximate polygon to check rectangularity
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)

                # Heuristic confidence based on aspect ratio proximity to standard ~3.5
                conf = max(0.40, min(0.85, 1.0 - abs(aspect_ratio - 3.5) / 4.0))

                # Ensure within image bounds
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w, x + cw)
                y2 = min(h, y + ch)

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(conf),
                    "method": "OpenCV Morphological / Contour"
                })

        # Apply Non-Maximum Suppression (NMS) to remove overlapping contours
        if detections:
            boxes = [d["bbox"] for d in detections]
            scores = [d["confidence"] for d in detections]
            indices = cv2.dnn.NMSBoxes(
                bboxes=[[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes],
                scores=scores,
                score_threshold=0.3,
                nms_threshold=0.4
            )
            if len(indices) > 0:
                final_detections = []
                for i in indices.flatten():
                    final_detections.append(detections[i])
                return final_detections

        return detections

    def detect(
        self,
        image: np.ndarray,
        mode: str = "auto",
        conf_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Detects license plates in image based on selected mode:
        - 'yolo': Deep learning YOLOv8 detector only
        - 'opencv': Traditional OpenCV contour pipeline only
        - 'auto' / 'ensemble': Try YOLOv8; if no detections found, fallback to OpenCV
        """
        if mode == "yolo":
            return self.detect_plates_yolo(image, conf_threshold=conf_threshold)
        elif mode == "opencv":
            return self.detect_plates_opencv(image)
        else: # Auto / Hybrid
            yolo_results = self.detect_plates_yolo(image, conf_threshold=conf_threshold)
            if yolo_results:
                return yolo_results
            return self.detect_plates_opencv(image)

    def get_pipeline_stages(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generates visual representations of every step in traditional computer vision
        pipeline for viva voce explanations and project demonstrations.
        """
        stages = {}

        # Stage 1: Input Original
        stages["1_original"] = image.copy()

        # Stage 2: Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        stages["2_grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Stage 3: Bilateral Filter (Noise reduction + Edge preservation)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        stages["3_bilateral_filter"] = cv2.cvtColor(bfilter, cv2.COLOR_GRAY2BGR)

        # Stage 4: Canny Edge Detection
        edged = cv2.Canny(bfilter, 30, 200)
        stages["4_canny_edges"] = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

        # Stage 5: Morphological Closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        stages["5_morphological_close"] = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

        # Stage 6: Contour Candidate Localization
        contour_viz = image.copy()
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        cv2.drawContours(contour_viz, contours, -1, (0, 255, 0), 2)
        stages["6_contours_located"] = contour_viz

        return stages
