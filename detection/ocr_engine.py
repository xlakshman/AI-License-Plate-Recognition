"""
detection/ocr_engine.py
Optical Character Recognition (OCR) Engine for Number Plates using EasyOCR and OpenCV Preprocessing.
Supports Indian HSRP, Bharat Series, and Custom formats with multi-pass recognition and heuristic correction.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import easyocr

from utils.preprocessing import preprocess_plate_for_ocr, save_detection_crop
from utils.validator import validate_plate, clean_plate_text, correct_ocr_mistakes_indian

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EASYOCR_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "easyocr")


class PlateOCREngine:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.model_dir = EASYOCR_MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        self.reader: Optional[easyocr.Reader] = None
        self._init_reader()

    def _init_reader(self):
        """Initializes EasyOCR reader with local model directory storage."""
        try:
            self.reader = easyocr.Reader(
                ['en'],
                gpu=self.use_gpu,
                model_storage_directory=self.model_dir,
                user_network_directory=self.model_dir,
                verbose=False
            )
            print("[PlateOCREngine] EasyOCR reader initialized successfully.")
        except Exception as e:
            print(f"[PlateOCREngine] Warning: EasyOCR reader initialization failed: {e}")
            self.reader = None

    def recognize_plate(
        self,
        plate_crop: np.ndarray,
        allowlist: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    ) -> Dict[str, Any]:
        """
        Runs multi-pass OCR recognition on plate crop:
        1. Evaluates contrast-enhanced crop
        2. Evaluates Otsu-binarized threshold
        3. Evaluates deskewed RGB crop
        4. Selects candidate with highest format validity and confidence
        """
        if plate_crop is None or plate_crop.size == 0 or self.reader is None:
            return {
                "plate_number": "UNKNOWN",
                "formatted_plate": "UNKNOWN",
                "confidence": 0.0,
                "is_valid": False,
                "state_code": "NA",
                "state_name": "Unknown",
                "format_type": "None",
                "raw_text": "",
                "crop_path": ""
            }

        prep = preprocess_plate_for_ocr(plate_crop)
        candidates: List[Dict[str, Any]] = []

        # List of image representations to test
        images_to_try = [
            ("contrast", prep.get("contrast")),
            ("deskewed", prep.get("deskewed")),
            ("thresh_otsu", prep.get("thresh_otsu")),
            ("thresh_adapt", prep.get("thresh_adapt")),
            ("thresh_inv", prep.get("thresh_inv"))
        ]

        for stage_name, img_variant in images_to_try:
            if img_variant is None:
                continue

            try:
                # Run EasyOCR
                results = self.reader.readtext(
                    img_variant,
                    detail=1,
                    allowlist=allowlist,
                    paragraph=False
                )

                if results:
                    # Combine all detected words horizontally
                    raw_str = " ".join([r[1] for r in results]).strip()
                    conf_scores = [float(r[2]) for r in results if len(r) > 2]
                    avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.5

                    cleaned = clean_plate_text(raw_str)
                    if len(cleaned) >= 4:
                        val_meta = validate_plate(cleaned)
                        score = avg_conf + (0.5 if val_meta["is_valid"] else 0.0)
                        candidates.append({
                            "raw_text": raw_str,
                            "plate_number": val_meta["plate_number"],
                            "formatted_plate": val_meta["formatted_plate"],
                            "confidence": round(avg_conf, 3),
                            "is_valid": val_meta["is_valid"],
                            "state_code": val_meta["state_code"],
                            "state_name": val_meta["state_name"],
                            "format_type": val_meta["format_type"],
                            "score": score
                        })
            except Exception as e:
                pass

        # Select candidate with best score
        if candidates:
            candidates.sort(key=lambda x: x["score"], reverse=True)
            best = candidates[0]
        else:
            # Fallback if EasyOCR detected nothing: apply default fallback
            best = {
                "raw_text": "",
                "plate_number": "UNKNOWN",
                "formatted_plate": "UNKNOWN",
                "confidence": 0.0,
                "is_valid": False,
                "state_code": "NA",
                "state_name": "Unknown",
                "format_type": "Unrecognized"
            }

        # Save crop image to disk
        crop_path = save_detection_crop(plate_crop, best["plate_number"])
        best["crop_path"] = crop_path

        return best
