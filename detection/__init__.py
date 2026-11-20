"""Detection package initialization."""
from detection.vehicle_detector import VehicleDetector, VEHICLE_CLASSES
from detection.plate_detector import PlateDetector
from detection.ocr_engine import PlateOCREngine
from detection.video_processor import VideoProcessor

__all__ = [
    "VehicleDetector",
    "VEHICLE_CLASSES",
    "PlateDetector",
    "PlateOCREngine",
    "VideoProcessor"
]
