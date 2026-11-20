"""
utils/preprocessing.py
Image Preprocessing and Computer Vision Enhancement Pipeline for License Plate Recognition.
Applies multi-stage image enhancement to optimize OCR text extraction.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional


def deskew_plate(image: np.ndarray) -> np.ndarray:
    """
    Detects plate inclination angle using Hough Lines and rotates to horizontal.
    Safely handles arbitrary HoughLinesP tensor dimensions.
    """
    if image is None or image.size == 0:
        return image

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 40,
            minLineLength=max(10, image.shape[1] // 4),
            maxLineGap=10
        )

        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                l = np.array(line).flatten()
                if len(l) >= 4:
                    x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(angle) < 25:  # Filter out vertical or extreme lines
                        angles.append(angle)

            if angles:
                median_angle = float(np.median(angles))
                if abs(median_angle) > 1.2:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(
                        image, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    return rotated
    except Exception as e:
        # Deskewing failure shouldn't crash the pipeline; fallback to original crop
        pass

    return image


def preprocess_plate_for_ocr(plate_crop: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Multi-stage computer vision enhancement on license plate crop:
    1. Standardize aspect/size (Upscale height to ~100px for optimal EasyOCR character stroke detection)
    2. Grayscale Conversion
    3. Bilateral Filter (Removes noise while preserving character boundaries)
    4. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
    5. Otsu's Adaptive Binarization
    6. Adaptive Gaussian Thresholding
    """
    if plate_crop is None or plate_crop.size == 0:
        return {}

    # Deskew first
    deskewed = deskew_plate(plate_crop)

    h, w = deskewed.shape[:2]
    if h == 0 or w == 0:
        return {}

    # Standardize height to 100px
    target_height = 100
    scale = target_height / float(h)
    new_w = max(50, int(w * scale))
    resized = cv2.resize(deskewed, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Bilateral Filter (Edge-preserving noise reduction)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # CLAHE for dynamic local contrast improvement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    # Otsu's binarization
    _, thresh_otsu = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive Gaussian thresholding (robust against uneven shadows and gradients)
    thresh_adapt = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
    )

    # Inverted versions
    thresh_otsu_inv = cv2.bitwise_not(thresh_otsu)

    return {
        "original_crop": plate_crop,
        "deskewed": deskewed,
        "resized": resized,
        "gray": gray,
        "contrast": contrast,
        "thresh_otsu": thresh_otsu,
        "thresh_adapt": thresh_adapt,
        "thresh_inv": thresh_otsu_inv
    }


def save_detection_crop(plate_crop: np.ndarray, plate_number: str, output_dir: str = "results/crops") -> str:
    """
    Saves the cropped plate image to disk and returns the relative path.
    """
    if plate_crop is None or plate_crop.size == 0:
        return ""

    os.makedirs(output_dir, exist_ok=True)
    clean_num = "".join(c for c in plate_number if c.isalnum()) or "UNKNOWN"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filename = f"crop_{clean_num}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    try:
        cv2.imwrite(filepath, plate_crop)
        return os.path.normpath(filepath).replace("\\", "/")
    except Exception as e:
        print(f"[Preprocessing] Error saving crop {filepath}: {e}")
        return ""
