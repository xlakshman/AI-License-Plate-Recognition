"""
src/video_processor.py
Video and Webcam Real-Time Processing Engine with Plate Stabilization,
Duplicate Suppression, and Database Logging.
"""

import time
from typing import Generator, Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from src.detector import PlateDetector
from src.ocr_engine import PlateOCREngine
from src.database import ALPRDatabase
from src.utils import draw_hud_bounding_box, save_detection_artifacts


class VideoStreamProcessor:
    def __init__(
        self,
        detector: Optional[PlateDetector] = None,
        ocr_engine: Optional[PlateOCREngine] = None,
        database: Optional[ALPRDatabase] = None,
        duplicate_cooldown_sec: float = 6.0
    ):
        self.detector = detector or PlateDetector()
        self.ocr_engine = ocr_engine or PlateOCREngine()
        self.database = database or ALPRDatabase()
        self.cooldown_sec = duplicate_cooldown_sec
        # Tracks {plate_number: last_seen_timestamp}
        self.recent_logged_plates: Dict[str, float] = {}

    def is_in_cooldown(self, plate_number: str) -> bool:
        """Determines if a plate was logged recently to prevent duplicate database spam."""
        clean = plate_number.replace(" ", "").upper()
        now = time.time()
        if clean in self.recent_logged_plates:
            if now - self.recent_logged_plates[clean] < self.cooldown_sec:
                return True
        return False

    def mark_logged(self, plate_number: str):
        """Updates last seen timestamp for a plate."""
        clean = plate_number.replace(" ", "").upper()
        self.recent_logged_plates[clean] = time.time()

    def process_frame(
        self,
        frame: np.ndarray,
        source_name: str = "Video",
        conf_threshold: float = 0.35,
        detection_mode: str = "auto",
        auto_log_db: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Processes a single video frame or webcam frame:
        1. Plate localization via YOLOv8 / OpenCV
        2. Cropping and OCR extraction
        3. Formatting, heuristic correction & validation
        4. Security status verification & auto-logging to database
        5. Visual HUD overlay rendering
        """
        detections = self.detector.detect(frame, mode=detection_mode, conf_threshold=conf_threshold)
        annotated_frame = frame.copy()
        frame_results = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            ocr_res = self.ocr_engine.recognize(plate_crop)
            plate_number = ocr_res["plate_number"]
            formatted_plate = ocr_res["formatted_plate"]
            confidence = ocr_res["confidence"]
            state_name = ocr_res["state_name"]
            state_code = ocr_res["state_code"]
            is_valid = ocr_res["is_valid"]

            # Query security status
            sec_info = self.database.check_security_status(plate_number)
            status = sec_info["status"]

            # Save and log to DB if valid and not in cooldown
            logged = False
            db_id = None
            if is_valid and plate_number != "UNREADABLE" and confidence >= 50.0:
                if auto_log_db and not self.is_in_cooldown(plate_number):
                    crop_rel, full_rel = save_detection_artifacts(annotated_frame, plate_crop, plate_number)
                    db_id = self.database.add_detection(
                        plate_number=plate_number,
                        raw_text=ocr_res["raw_text"],
                        confidence=confidence,
                        state_code=state_code,
                        state_name=state_name,
                        vehicle_type="Vehicle",
                        source=source_name,
                        crop_image_path=crop_rel,
                        full_image_path=full_rel
                    )
                    self.mark_logged(plate_number)
                    logged = True

            # Draw HUD overlay
            annotated_frame = draw_hud_bounding_box(
                annotated_frame,
                det["bbox"],
                formatted_plate,
                confidence,
                status=status,
                state_name=state_name,
                method=det["method"]
            )

            frame_results.append({
                "bbox": det["bbox"],
                "plate_number": plate_number,
                "formatted_plate": formatted_plate,
                "confidence": confidence,
                "state_name": state_name,
                "status": status,
                "is_valid": is_valid,
                "logged": logged,
                "db_id": db_id,
                "crop": plate_crop
            })

        return annotated_frame, frame_results

    def process_video_file(
        self,
        video_path: str,
        frame_skip: int = 2,
        conf_threshold: float = 0.35,
        detection_mode: str = "auto"
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yields processed frames and stats generator for video file playback in UI.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        all_unique_plates = set()

        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            curr_time = time.time()
            elapsed = curr_time - prev_time
            curr_fps = (1.0 / elapsed) if elapsed > 0 else 0
            prev_time = curr_time

            annotated, results = self.process_frame(
                frame,
                source_name="Video File",
                conf_threshold=conf_threshold,
                detection_mode=detection_mode,
                auto_log_db=True
            )

            for r in results:
                if r["is_valid"] and r["plate_number"] != "UNREADABLE":
                    all_unique_plates.add(r["formatted_plate"])

            yield {
                "frame_idx": frame_idx,
                "total_frames": total_frames,
                "progress": min(1.0, frame_idx / max(1, total_frames)),
                "fps": round(curr_fps, 1),
                "annotated_frame": annotated,
                "detections": results,
                "unique_plates_count": len(all_unique_plates),
                "unique_plates_list": list(all_unique_plates)
            }

        cap.release()
