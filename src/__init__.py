"""
src Package Initialization
AI-Based Automatic License Plate Detection and Recognition System (ALPR)
"""

from src.detector import PlateDetector
from src.ocr_engine import PlateOCREngine
from src.validator import validate_plate, clean_plate_text, INDIAN_STATES
from src.database import ALPRDatabase
from src.video_processor import VideoStreamProcessor
from src.utils import draw_hud_bounding_box, save_detection_artifacts, compute_iou

__all__ = [
    "PlateDetector",
    "PlateOCREngine",
    "validate_plate",
    "clean_plate_text",
    "INDIAN_STATES",
    "ALPRDatabase",
    "VideoStreamProcessor",
    "draw_hud_bounding_box",
    "save_detection_artifacts",
    "compute_iou"
]
