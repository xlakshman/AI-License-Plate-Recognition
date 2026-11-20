"""
src/utils.py
Visualization utilities, HUD drawing, file artifact managers, and evaluation metrics.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any, List

CROPS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "crops")
os.makedirs(CROPS_DIR, exist_ok=True)


def draw_hud_bounding_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    plate_text: str,
    confidence: float,
    status: str = "Allowed",
    state_name: str = "",
    method: str = ""
) -> np.ndarray:
    """
    Renders an elegant, modern Computer Vision HUD overlay around the detected license plate.
    Features:
    - Status-coded color palette (Green = Allowed/Whitelisted, Red = Flagged, Blue = Visitor)
    - Corner brackets and translucent header badge
    - State indicator and confidence badge
    """
    annotated = image.copy()
    x1, y1, x2, y2 = bbox

    # Color scheme based on security status
    if "Flagged" in status or "Blacklisted" in status:
        primary_color = (40, 40, 220)       # Crimson Red (BGR)
        bg_color = (20, 20, 160)
        badge_text = "FLAGGED / BLACKLISTED"
    elif "Whitelisted" in status or "Allowed" in status:
        primary_color = (60, 200, 60)       # Emerald Green (BGR)
        bg_color = (30, 140, 30)
        badge_text = "AUTHORIZED / ALLOWED"
    else:
        primary_color = (230, 150, 30)      # Electric Blue / Cyan
        bg_color = (180, 100, 20)
        badge_text = "VISITOR VEHICLE"

    # 1. Bounding rectangle outline
    cv2.rectangle(annotated, (x1, y1), (x2, y2), primary_color, 2)

    # 2. Corner brackets for futuristic HUD aesthetic
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    # Top-Left
    cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), primary_color, 4)
    cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), primary_color, 4)
    # Top-Right
    cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), primary_color, 4)
    cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), primary_color, 4)
    # Bottom-Left
    cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), primary_color, 4)
    cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), primary_color, 4)
    # Bottom-Right
    cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), primary_color, 4)
    cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), primary_color, 4)

    # 3. Label text assembly
    label_main = plate_text if plate_text and plate_text != "UNREADABLE" else "DETECTING..."
    label_sub = f"{state_name} | {confidence:.1f}%" if state_name and state_name != "Unknown" else f"{confidence:.1f}%"

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale_main = 0.65
    font_scale_sub = 0.45
    thickness = 1

    (tw1, th1), _ = cv2.getTextSize(label_main, font, font_scale_main, thickness + 1)
    (tw2, th2), _ = cv2.getTextSize(label_sub, font, font_scale_sub, thickness)
    badge_w = max(tw1, tw2, 140) + 20
    badge_h = th1 + th2 + 20

    # Draw label above or below depending on space
    by2 = y1 - 8
    by1 = max(0, by2 - badge_h)
    bx1 = x1
    bx2 = min(image.shape[1], bx1 + badge_w)

    # If top runs out of bounds, place below
    if by1 < 5:
        by1 = y2 + 8
        by2 = min(image.shape[0], by1 + badge_h)

    # Semi-transparent label background overlay
    overlay = annotated.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    # Label border
    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), primary_color, 1)

    # Text rendering
    cv2.putText(annotated, label_main, (bx1 + 8, by1 + th1 + 4), font, font_scale_main, (255, 255, 255), thickness + 1, cv2.LINE_AA)
    cv2.putText(annotated, label_sub, (bx1 + 8, by1 + th1 + th2 + 10), font, font_scale_sub, (200, 240, 255), thickness, cv2.LINE_AA)

    return annotated


def save_detection_artifacts(
    full_image: np.ndarray,
    crop_image: np.ndarray,
    plate_number: str
) -> Tuple[str, str]:
    """
    Saves cropped license plate and full annotated image to disk.
    Returns relative paths suitable for storage in SQLite database.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    safe_plate = "".join([c if c.isalnum() else "_" for c in plate_number])

    crop_filename = f"crop_{safe_plate}_{timestamp}.png"
    full_filename = f"full_{safe_plate}_{timestamp}.png"

    crop_rel_path = os.path.join("data", "crops", crop_filename)
    full_rel_path = os.path.join("data", "crops", full_filename)

    crop_abs_path = os.path.join(CROPS_DIR, crop_filename)
    full_abs_path = os.path.join(CROPS_DIR, full_filename)

    if crop_image is not None and crop_image.size > 0:
        cv2.imwrite(crop_abs_path, crop_image)
    if full_image is not None and full_image.size > 0:
        cv2.imwrite(full_abs_path, full_image)

    return crop_rel_path, full_rel_path


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    Standard evaluation metric for object detection.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom
