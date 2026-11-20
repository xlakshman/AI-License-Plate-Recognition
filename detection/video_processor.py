"""
detection/video_processor.py
Video Processing Engine for ANPR System.
Processes video streams frame-by-frame, performs vehicle & license plate detection,
applies duplicate suppression across consecutive frames, and renders annotated video output.
"""

import os
import time
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from detection.vehicle_detector import VehicleDetector
from detection.plate_detector import PlateDetector
from detection.ocr_engine import PlateOCREngine
from utils.visualization import annotate_frame
from utils.validator import clean_plate_text
from database.db import insert_detection


def compute_string_similarity(s1: str, s2: str) -> float:
    """Computes basic character matching ratio between two plate strings."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    # Levenshtein distance
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    dist = dp[m][n]
    max_len = max(m, n)
    return 1.0 - (dist / float(max_len)) if max_len > 0 else 0.0


class VideoProcessor:
    def __init__(
        self,
        vehicle_detector: Optional[VehicleDetector] = None,
        plate_detector: Optional[PlateDetector] = None,
        ocr_engine: Optional[PlateOCREngine] = None
    ):
        self.vehicle_detector = vehicle_detector or VehicleDetector()
        self.plate_detector = plate_detector or PlateDetector()
        self.ocr_engine = ocr_engine or PlateOCREngine()

    def process_video(
        self,
        video_path: str,
        output_dir: str = "results/videos",
        frame_skip: int = 3,
        conf_threshold: float = 0.30,
        max_frames: int = 400
    ) -> Dict[str, Any]:
        """
        Processes video frame by frame:
        - Detects vehicles and license plates.
        - Suppresses duplicate detections across consecutive frames.
        - Writes annotated output video.
        - Persists unique plate records to database.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_{timestamp_str}.mp4"
        output_filepath = os.path.join(output_dir, output_filename)

        # Video writer using mp4v or avc1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filepath, fourcc, fps / max(1, frame_skip), (width, height))

        frame_idx = 0
        processed_count = 0
        start_time = time.time()

        # Duplicate suppression tracking dictionary
        # key: normalized plate, value: dict with best detection info
        tracked_records: Dict[str, Dict[str, Any]] = {}

        last_vehicles: List[Dict[str, Any]] = []
        last_plates: List[Dict[str, Any]] = []

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and processed_count >= max_frames):
                break

            # Process every N-th frame for high performance
            if frame_idx % frame_skip == 0:
                processed_count += 1

                # 1. Detect vehicles
                vehicles = self.vehicle_detector.detect_vehicles(frame, conf_threshold=conf_threshold)

                # 2. Detect license plates
                plates = self.plate_detector.detect_plates(frame, conf_threshold=conf_threshold)

                # 3. OCR and match with vehicle
                annotated_plates = []
                for p in plates:
                    crop = p.get("crop")
                    ocr_res = self.ocr_engine.recognize_plate(crop)

                    plate_num = ocr_res.get("plate_number", "")
                    clean_num = clean_plate_text(plate_num)

                    if clean_num and clean_num != "UNKNOWN" and len(clean_num) >= 5:
                        p["plate_number"] = ocr_res.get("formatted_plate", clean_num)
                        p["confidence"] = ocr_res.get("confidence", 0.0)
                        annotated_plates.append(p)

                        # Find matching vehicle based on bounding box containment
                        px1, py1, px2, py2 = p["bbox"]
                        matched_vtype = "Car"
                        matched_vconf = 0.85
                        for v in vehicles:
                            vx1, vy1, vx2, vy2 = v["bbox"]
                            if vx1 <= px1 and vy1 <= py1 and vx2 >= px2 and vy2 >= py2:
                                matched_vtype = v["class_name"]
                                matched_vconf = v["confidence"]
                                break

                        # Check duplicate suppression against tracked records
                        is_duplicate = False
                        matched_key = None

                        for key, track in tracked_records.items():
                            sim = compute_string_similarity(clean_num, key)
                            frames_since = frame_idx - track["last_seen_frame"]

                            # If same plate within 90 frames (~3 sec) or very high similarity
                            if sim >= 0.85 and frames_since < 90:
                                is_duplicate = True
                                matched_key = key
                                break

                        if is_duplicate and matched_key:
                            # Update existing track with best confidence
                            track = tracked_records[matched_key]
                            track["last_seen_frame"] = frame_idx
                            track["total_detections"] += 1
                            if ocr_res.get("confidence", 0.0) > track["confidence"]:
                                track["confidence"] = ocr_res["confidence"]
                                track["crop_path"] = ocr_res.get("crop_path", track["crop_path"])
                                track["plate_number"] = ocr_res.get("formatted_plate", track["plate_number"])
                        else:
                            # Genuine new vehicle detection
                            tracked_records[clean_num] = {
                                "plate_number": ocr_res.get("formatted_plate", clean_num),
                                "raw_text": ocr_res.get("raw_text", clean_num),
                                "vehicle_type": matched_vtype,
                                "confidence": ocr_res.get("confidence", 0.75),
                                "vehicle_confidence": matched_vconf,
                                "state_code": ocr_res.get("state_code", "UNKNOWN"),
                                "state_name": ocr_res.get("state_name", "Unknown State"),
                                "is_valid": 1 if ocr_res.get("is_valid", False) else 0,
                                "format_type": ocr_res.get("format_type", "Standard"),
                                "crop_path": ocr_res.get("crop_path", ""),
                                "image_path": output_filepath.replace("\\", "/"),
                                "source_type": "video",
                                "source_name": os.path.basename(video_path),
                                "first_seen_frame": frame_idx,
                                "last_seen_frame": frame_idx,
                                "total_detections": 1
                            }

                last_vehicles = vehicles
                last_plates = annotated_plates

                # Annotate current frame
                annotated = annotate_frame(frame, last_vehicles, last_plates)
                out.write(annotated)

            frame_idx += 1

        cap.release()
        out.release()
        elapsed_time = round(time.time() - start_time, 2)

        # Save unique suppressed detections into SQLite database
        unique_results = []
        for key, rec in tracked_records.items():
            db_id = insert_detection(rec)
            rec["id"] = db_id
            unique_results.append(rec)

        return {
            "output_video": output_filepath.replace("\\", "/"),
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "fps": fps,
            "elapsed_time": elapsed_time,
            "unique_detections_count": len(unique_results),
            "detections": unique_results
        }
