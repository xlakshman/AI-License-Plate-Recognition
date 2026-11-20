"""
utils/visualization.py
Visualization utilities for annotating vehicles, license plates, and OCR results.
Features modern HUD style bounding boxes, side-by-side comparison generator, and high-visibility badges.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple


# Color Palette (BGR)
COLOR_VEHICLE = (46, 204, 113)     # Emerald Green
COLOR_PLATE = (0, 165, 255)        # Electric Amber / Orange
COLOR_TEXT_BG = (15, 23, 42)       # Dark Navy #0F172A
COLOR_TEXT = (255, 255, 255)       # White
COLOR_CORNER = (0, 225, 255)       # Cyan accent for HUD corners


def draw_hud_box(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int] = COLOR_PLATE,
    line_thickness: int = 2,
    corner_length: int = 15
) -> np.ndarray:
    """
    Draws a cyber/modern HUD styled bounding box with highlighted corners and sleek label pill.
    """
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Base bounding box with slight alpha or solid
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)

    # Dynamic corner length based on box dimension
    cw = min(corner_length, (x2 - x1) // 3)
    ch = min(corner_length, (y2 - y1) // 3)

    corner_color = COLOR_CORNER
    cthick = line_thickness + 1

    # Top-Left corner
    cv2.line(img, (x1, y1), (x1 + cw, y1), corner_color, cthick, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + ch), corner_color, cthick, cv2.LINE_AA)

    # Top-Right corner
    cv2.line(img, (x2, y1), (x2 - cw, y1), corner_color, cthick, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + ch), corner_color, cthick, cv2.LINE_AA)

    # Bottom-Left corner
    cv2.line(img, (x1, y2), (x1 + cw, y2), corner_color, cthick, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - ch), corner_color, cthick, cv2.LINE_AA)

    # Bottom-Right corner
    cv2.line(img, (x2, y2), (x2 - cw, y2), corner_color, cthick, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - ch), corner_color, cthick, cv2.LINE_AA)

    # Label pill
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Place label above the box if room allows, else inside
        if y1 - th - 12 >= 0:
            ly1 = y1 - th - 12
            ly2 = y1
        else:
            ly1 = y1
            ly2 = y1 + th + 12

        lx1 = x1
        lx2 = x1 + tw + 14

        # Background pill
        cv2.rectangle(img, (lx1, ly1), (lx2, ly2), COLOR_TEXT_BG, -1)
        cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, 1, cv2.LINE_AA)

        # Text
        cv2.putText(
            img, label, (lx1 + 7, ly2 - 6),
            font, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA
        )

    return img


def annotate_frame(
    image: np.ndarray,
    vehicle_detections: List[Dict[str, Any]],
    plate_detections: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Draws both vehicle bounding boxes and license plate bounding boxes on the image.
    """
    annotated = image.copy()

    # 1. Draw vehicle bounding boxes
    for v in vehicle_detections:
        box = v.get("bbox", (0, 0, 0, 0))
        label_name = v.get("class_name", "Vehicle").capitalize()
        conf = v.get("confidence", 0.0)
        label = f"{label_name} {int(conf * 100)}%"
        draw_hud_box(annotated, box, label, color=COLOR_VEHICLE, line_thickness=2)

    # 2. Draw plate bounding boxes
    for p in plate_detections:
        box = p.get("bbox", (0, 0, 0, 0))
        plate_str = p.get("plate_number") or p.get("raw_text") or "PLATE"
        conf = p.get("confidence", 0.0)
        label = f"[{plate_str}] {int(conf * 100)}%" if conf > 0 else f"[{plate_str}]"
        draw_hud_box(annotated, box, label, color=COLOR_PLATE, line_thickness=2)

    return annotated


def create_side_by_side(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """
    Constructs a side-by-side comparison image with header banners for UI display.
    """
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]

    # Target height
    target_h = max(h1, h2, 480)
    # Rescale while maintaining aspect ratio
    scale1 = target_h / float(h1)
    new_w1 = int(w1 * scale1)
    img1 = cv2.resize(original, (new_w1, target_h), interpolation=cv2.INTER_AREA)

    scale2 = target_h / float(h2)
    new_w2 = int(w2 * scale2)
    img2 = cv2.resize(processed, (new_w2, target_h), interpolation=cv2.INTER_AREA)

    # Add header banner on top
    header_h = 45
    canvas = np.zeros((target_h + header_h, new_w1 + new_w2, 3), dtype=np.uint8)
    canvas[:header_h, :] = (20, 24, 33)  # Dark navy banner

    canvas[header_h:, :new_w1] = img1
    canvas[header_h:, new_w1:] = img2

    # Draw separator line
    cv2.line(canvas, (new_w1, 0), (new_w1, target_h + header_h), (50, 60, 80), 2)

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "ORIGINAL IMAGE", (20, 30), font, 0.7, (200, 210, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, "AI ANPR DETECTION & OCR OUTPUT", (new_w1 + 20, 30), font, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

    return canvas


def save_annotated_image(image: np.ndarray, prefix: str = "result", output_dir: str = "results/images") -> str:
    """
    Saves an annotated image to disk and returns the relative path.
    """
    if image is None or image.size == 0:
        return ""

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)

    try:
        cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return os.path.normpath(filepath).replace("\\", "/")
    except Exception as e:
        print(f"[Visualization] Error saving image {filepath}: {e}")
        return ""
