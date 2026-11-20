"""
src/ocr_engine.py
License Plate Optical Character Recognition (OCR) Engine using EasyOCR and OpenCV Preprocessing.
Features:
- Multi-stage image enhancement (Bicubic upscaling, CLAHE contrast tuning, Bilateral denoising, Otsu thresholding)
- Skew angle detection and deskewing
- Multi-pass OCR candidate evaluation
- Character disambiguation and format validation via src.validator
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import easyocr
from src.validator import validate_plate, clean_plate_text


class PlateOCREngine:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.reader: Optional[easyocr.Reader] = None
        self._init_reader()

    def _init_reader(self):
        """Initializes EasyOCR reader with English alphanumeric focus."""
        try:
            self.reader = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)
        except Exception as e:
            print(f"[OCR] Warning initializing EasyOCR: {e}")
            self.reader = None

    def preprocess_plate(self, plate_crop: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Executes multi-stage computer vision enhancement on license plate crop:
        1. Bicubic Upscaling (Standardizing height to 100px for optimal OCR receptive field)
        2. Grayscale Conversion
        3. Bilateral Filter (Reduces noise while preserving character boundary sharpness)
        4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        5. Otsu's Adaptive Binarization
        """
        h, w = plate_crop.shape[:2]
        if h == 0 or w == 0:
            return {}

        # 1. Upscale if too small
        target_height = 100
        scale = target_height / float(h)
        new_w = int(w * scale)
        resized = cv2.resize(plate_crop, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

        # 2. Grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 3. Bilateral Filter
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # 4. CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)

        # 5. Otsu thresholding
        _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Also create inverted threshold (dark text on light bg or vice-versa)
        thresh_inv = cv2.bitwise_not(thresh)

        return {
            "resized": resized,
            "gray": gray,
            "contrast": contrast,
            "thresh": thresh,
            "thresh_inv": thresh_inv
        }

    def deskew_plate(self, image: np.ndarray) -> np.ndarray:
        """
        Detects plate inclination angle and deskews the image horizontally.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=image.shape[1] // 3, maxLineGap=10)

        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 30:  # Only consider subtle skews
                    angles.append(angle)

            if angles:
                median_angle = float(np.median(angles))
                if abs(median_angle) > 1.0:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(
                        image, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    return rotated

        return image

    def recognize(self, plate_crop: np.ndarray) -> Dict[str, Any]:
        """
        Performs Optical Character Recognition on cropped license plate.
        Evaluates enhanced image variants and picks best candidate matching vehicle format.
        """
        if self.reader is None or plate_crop is None or plate_crop.size == 0:
            return {
                "plate_number": "UNREADABLE",
                "formatted_plate": "UNREADABLE",
                "raw_text": "",
                "confidence": 0.0,
                "is_valid": False,
                "state_name": "Unknown",
                "state_code": "N/A",
                "format_type": "None",
                "preprocessed_images": {}
            }

        # Multi-stage enhancement
        preprocessed = self.preprocess_plate(plate_crop)
        if not preprocessed:
            return {
                "plate_number": "UNREADABLE",
                "formatted_plate": "UNREADABLE",
                "raw_text": "",
                "confidence": 0.0,
                "is_valid": False,
                "state_name": "Unknown",
                "state_code": "N/A",
                "format_type": "None",
                "preprocessed_images": {}
            }

        candidates = []

        # Candidate 1: Raw resized crop
        raw_res = self.reader.readtext(
            preprocessed["resized"],
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        )
        for _, text, conf in raw_res:
            candidates.append((text, conf, "RGB Resized"))

        # Candidate 2: Contrast enhanced
        contrast_res = self.reader.readtext(
            preprocessed["contrast"],
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        )
        for _, text, conf in contrast_res:
            candidates.append((text, conf, "CLAHE Contrast"))

        # Candidate 3: Otsu Threshold
        thresh_res = self.reader.readtext(
            preprocessed["thresh"],
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        )
        for _, text, conf in thresh_res:
            candidates.append((text, conf, "Otsu Threshold"))

        if not candidates:
            # Fallback without allowlist
            fallback_res = self.reader.readtext(plate_crop)
            for _, text, conf in fallback_res:
                candidates.append((text, conf, "Original Fallback"))

        # Rank candidates using validation heuristics and confidence
        best_candidate = None
        best_score = -1.0
        best_validation: Dict[str, Any] = {}

        for raw_text, conf, src_step in candidates:
            val_info = validate_plate(raw_text)
            clean_len = len(val_info["plate_number"])

            # Score computation:
            # Valid standard format gets large bonus (+2.0)
            # Correct length (9-10 chars) gets bonus (+1.0)
            # EasyOCR confidence weight (0.0 to 1.0)
            score = float(conf)
            if val_info["is_valid"]:
                score += 2.0
            if 8 <= clean_len <= 11:
                score += 1.0

            if score > best_score:
                best_score = score
                best_candidate = raw_text
                best_validation = val_info
                best_conf = float(conf)

        if best_candidate and best_validation:
            confidence_pct = round(min(100.0, best_conf * 100.0), 2)
            return {
                "plate_number": best_validation["plate_number"],
                "formatted_plate": best_validation["formatted_plate"],
                "raw_text": best_candidate,
                "confidence": confidence_pct,
                "is_valid": best_validation["is_valid"],
                "state_code": best_validation.get("state_code", "N/A"),
                "state_name": best_validation.get("state_name", "Unknown"),
                "format_type": best_validation.get("format_type", "Standard"),
                "preprocessed_images": preprocessed
            }

        return {
            "plate_number": "UNREADABLE",
            "formatted_plate": "UNREADABLE",
            "raw_text": "",
            "confidence": 0.0,
            "is_valid": False,
            "state_name": "Unknown",
            "state_code": "N/A",
            "format_type": "None",
            "preprocessed_images": preprocessed
        }
