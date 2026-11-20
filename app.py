"""
app.py
AI-Based Automatic License Plate Detection and Recognition System (ANPR)
Flask Application Server for B.Tech CSE (AI & ML) College Project and Viva Examination.
"""

import os
import sys

# Configure environment limits BEFORE importing heavy libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["YOLO_VERBOSE"] = "False"

import time
import cv2
import numpy as np
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify, send_file,
    send_from_directory, Response, redirect, url_for
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.db import (
    init_db, insert_detection, get_recent_detections, get_all_detections,
    count_detections, delete_detection, clear_all_detections, get_statistics,
    get_vehicle_distribution, get_daily_analytics, get_state_distribution,
    export_to_csv_string
)
from utils.visualization import (
    annotate_frame, create_side_by_side, save_annotated_image
)
from utils.validator import clean_plate_text

# Initialize Flask App
app = Flask(__name__)
app.config["SECRET_KEY"] = "anpr-ai-ml-project-secret-key-2026"
app.config["UPLOAD_FOLDER"] = os.path.join(PROJECT_ROOT, "uploads")
app.config["RESULTS_FOLDER"] = os.path.join(PROJECT_ROOT, "results")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB maximum upload

# Ensure directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(os.path.join(app.config["RESULTS_FOLDER"], "images"), exist_ok=True)
os.makedirs(os.path.join(app.config["RESULTS_FOLDER"], "crops"), exist_ok=True)
os.makedirs(os.path.join(app.config["RESULTS_FOLDER"], "videos"), exist_ok=True)

# Initialize Database
init_db()

# Lazy-loaded AI Engine singletons (loaded on first request to stay within 512MB free-tier RAM)
_vehicle_detector = None
_plate_detector = None
_ocr_engine = None
_video_processor = None

def get_ai_engine():
    """Lazy-load AI models only when first inference request arrives (preserves RAM on 512MB hosts)."""
    global _vehicle_detector, _plate_detector, _ocr_engine, _video_processor
    if _vehicle_detector is None:
        print("[System] Lazy-loading AI Detection & OCR Models...")
        import gc
        import torch
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)

        from detection.vehicle_detector import VehicleDetector
        from detection.plate_detector import PlateDetector
        from detection.ocr_engine import PlateOCREngine
        from detection.video_processor import VideoProcessor

        _vehicle_detector = VehicleDetector()
        _plate_detector = PlateDetector()
        _ocr_engine = PlateOCREngine()
        _video_processor = VideoProcessor(_vehicle_detector, _plate_detector, _ocr_engine)
        gc.collect()
        print("[System] AI Engine Ready and memory optimized.")
    return _vehicle_detector, _plate_detector, _ocr_engine, _video_processor


# ---------------------------------------------------------
# Static File Routes for Processed Media & Samples
# ---------------------------------------------------------
@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)


@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/data/samples/<path:filename>")
def serve_samples(filename):
    samples_dir = os.path.join(PROJECT_ROOT, "data", "samples")
    return send_from_directory(samples_dir, filename)


# ---------------------------------------------------------
# HTML Frontend Page Routes
# ---------------------------------------------------------
@app.route("/")
def dashboard():
    """Renders the executive dashboard with metrics, charts, and recent records."""
    stats = get_statistics()
    recent = get_recent_detections(limit=8)
    return render_template("dashboard.html", stats=stats, recent=recent)


@app.route("/image")
def image_detection():
    """Renders image detection workspace with file dropzone and sample selector."""
    sample_dir = os.path.join(PROJECT_ROOT, "data", "samples", "cars")
    samples = []
    if os.path.exists(sample_dir):
        for f in os.listdir(sample_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append(f)
    return render_template("image_detection.html", samples=samples)


@app.route("/video")
def video_detection():
    """Renders video detection workspace with upload and sample video options."""
    sample_vids_dir = os.path.join(PROJECT_ROOT, "data", "samples", "videos")
    samples = []
    if os.path.exists(sample_vids_dir):
        for f in os.listdir(sample_vids_dir):
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                samples.append(f)
    return render_template("video_detection.html", samples=samples)


@app.route("/history")
def history():
    """Renders detection history table with search and filtering."""
    return render_template("history.html")


@app.route("/analytics")
def analytics():
    """Renders advanced visual analytics dashboard."""
    stats = get_statistics()
    return render_template("analytics.html", stats=stats)


@app.route("/about")
def about():
    """Renders academic project documentation, viva preparation, and system specs."""
    return render_template("about.html")


# ---------------------------------------------------------
# RESTful API Endpoints
# ---------------------------------------------------------
@app.route("/api/detect/image", methods=["POST"])
def api_detect_image():
    """
    Processes an uploaded image or sample image:
    Vehicle Detection -> Plate Detection -> Cropping -> Preprocessing -> OCR -> Validation -> Database.
    """
    start_time = time.time()
    image_path = None
    filename = ""

    # Check whether file was uploaded or sample was chosen
    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(image_path)
    elif "sample" in request.form and request.form["sample"]:
        sample_name = request.form["sample"]
        sample_path = os.path.join(PROJECT_ROOT, "data", "samples", "cars", sample_name)
        if os.path.exists(sample_path):
            image_path = sample_path
            filename = sample_name
        else:
            return jsonify({"success": False, "error": f"Sample image {sample_name} not found"}), 404
    else:
        return jsonify({"success": False, "error": "No image file or sample provided"}), 400

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"success": False, "error": "Unable to decode image"}), 400

    conf_thresh = float(request.form.get("confidence", 0.25))
    method = request.form.get("method", "auto")

    # Lazy-load AI models on first request
    vehicle_detector, plate_detector, ocr_engine, video_processor = get_ai_engine()

    # 1. License Plate Detection
    plate_detections = plate_detector.detect_plates(img, method=method, conf_threshold=conf_thresh)
    plate_boxes = [p["bbox"] for p in plate_detections]

    # 2. Vehicle Detection
    vehicle_detections = vehicle_detector.detect_vehicles(img, conf_threshold=conf_thresh, plate_boxes=plate_boxes)

    # 3. Plate OCR & Formatting
    results_list = []
    for p in plate_detections:
        crop = p.get("crop")
        ocr_res = ocr_engine.recognize_plate(crop)

        # Match with vehicle bounding box
        px1, py1, px2, py2 = p["bbox"]
        matched_vtype = "Car"
        matched_vconf = 0.85
        for v in vehicle_detections:
            vx1, vy1, vx2, vy2 = v["bbox"]
            if vx1 <= px1 and vy1 <= py1 and vx2 >= px2 and vy2 >= py2:
                matched_vtype = v["class_name"]
                matched_vconf = v["confidence"]
                break

        p["plate_number"] = ocr_res.get("formatted_plate", "UNKNOWN")
        p["confidence"] = ocr_res.get("confidence", 0.0)

        record_data = {
            "plate_number": ocr_res.get("formatted_plate", "UNKNOWN"),
            "raw_text": ocr_res.get("raw_text", ""),
            "vehicle_type": matched_vtype,
            "confidence": ocr_res.get("confidence", 0.0),
            "vehicle_confidence": matched_vconf,
            "state_code": ocr_res.get("state_code", "UNKNOWN"),
            "state_name": ocr_res.get("state_name", "Unknown State"),
            "is_valid": 1 if ocr_res.get("is_valid", False) else 0,
            "format_type": ocr_res.get("format_type", "Standard"),
            "crop_path": ocr_res.get("crop_path", ""),
            "source_type": "image",
            "source_name": filename,
            "bbox": p["bbox"],
            "detection_method": p.get("method", "YOLOv8")
        }
        results_list.append(record_data)

    # 4. Generate Annotated Image
    annotated_img = annotate_frame(img, vehicle_detections, plate_detections)
    annotated_path = save_annotated_image(annotated_img, prefix="anpr_annotated")

    # 5. Generate Side-by-Side Comparison Image
    side_by_side = create_side_by_side(img, annotated_img)
    side_by_side_path = save_annotated_image(side_by_side, prefix="comparison")

    # 6. Save detections to Database
    for rec in results_list:
        rec["image_path"] = annotated_path
        db_id = insert_detection(rec)
        rec["id"] = db_id

    elapsed_time = round(time.time() - start_time, 2)

    return jsonify({
        "success": True,
        "filename": filename,
        "elapsed_time": elapsed_time,
        "vehicles_count": len(vehicle_detections),
        "plates_count": len(plate_detections),
        "vehicles": vehicle_detections,
        "detections": results_list,
        "annotated_image": "/" + annotated_path if annotated_path else "",
        "comparison_image": "/" + side_by_side_path if side_by_side_path else "",
        "original_image": f"/uploads/{filename}" if "upload_" in filename else f"/data/samples/cars/{filename}"
    })


@app.route("/api/detect/video", methods=["POST"])
def api_detect_video():
    """
    Processes video frame-by-frame with duplicate suppression across frames.
    """
    video_path = None
    filename = ""

    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        filename = f"upload_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(video_path)
    elif "sample" in request.form and request.form["sample"]:
        sample_name = request.form["sample"]
        sample_path = os.path.join(PROJECT_ROOT, "data", "samples", "videos", sample_name)
        if os.path.exists(sample_path):
            video_path = sample_path
            filename = sample_name
        else:
            return jsonify({"success": False, "error": f"Sample video {sample_name} not found"}), 404
    else:
        return jsonify({"success": False, "error": "No video file or sample provided"}), 400

    conf_thresh = float(request.form.get("confidence", 0.30))
    frame_skip = int(request.form.get("frame_skip", 3))

    try:
        _, _, _, video_processor = get_ai_engine()
        proc_result = video_processor.process_video(
            video_path=video_path,
            output_dir=os.path.join(app.config["RESULTS_FOLDER"], "videos"),
            frame_skip=frame_skip,
            conf_threshold=conf_thresh
        )

        return jsonify({
            "success": True,
            "filename": filename,
            "output_video": "/" + proc_result["output_video"],
            "total_frames": proc_result["total_frames"],
            "processed_frames": proc_result["processed_frames"],
            "fps": proc_result["fps"],
            "elapsed_time": proc_result["elapsed_time"],
            "unique_detections_count": proc_result["unique_detections_count"],
            "detections": proc_result["detections"]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """Fetches paginated and filtered detection records."""
    search = request.args.get("search", "")
    vehicle_type = request.args.get("vehicle_type", "all")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    is_valid_param = request.args.get("is_valid", None)

    is_valid = None
    if is_valid_param in ["1", "true", "True"]:
        is_valid = 1
    elif is_valid_param in ["0", "false", "False"]:
        is_valid = 0

    page = max(1, int(request.args.get("page", 1)))
    limit = max(5, min(100, int(request.args.get("limit", 15))))
    offset = (page - 1) * limit

    total = count_detections(
        search=search, vehicle_type=vehicle_type,
        date_from=date_from, date_to=date_to, is_valid=is_valid
    )
    records = get_all_detections(
        search=search, vehicle_type=vehicle_type,
        date_from=date_from, date_to=date_to, is_valid=is_valid,
        limit=limit, offset=offset
    )

    total_pages = (total + limit - 1) // limit if total > 0 else 1

    return jsonify({
        "success": True,
        "records": records,
        "total": total,
        "page": page,
        "pages": total_pages,
        "limit": limit
    })


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def api_delete_record(record_id):
    """Deletes a specific detection record."""
    deleted = delete_detection(record_id)
    return jsonify({"success": deleted})


@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    """Clears all detection records from database."""
    clear_all_detections()
    return jsonify({"success": True})


@app.route("/api/export/csv")
def api_export_csv():
    """Exports all detection records to a CSV file."""
    csv_data = export_to_csv_string()
    filename = f"anpr_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )


@app.route("/api/stats")
def api_stats():
    """Returns analytics data for dynamic Chart.js rendering."""
    stats = get_statistics()
    v_dist = get_vehicle_distribution()
    daily = get_daily_analytics(days=7)
    states = get_state_distribution(limit=5)

    return jsonify({
        "stats": stats,
        "vehicle_distribution": v_dist,
        "daily_analytics": daily,
        "state_distribution": states
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================================")
    print(f"  AI-BASED AUTOMATIC LICENSE PLATE RECOGNITION (ANPR) SYSTEM")
    print(f"  Server running at http://127.0.0.1:{port}")
    print(f"  Designed for B.Tech CSE (AI & ML) College Project & Viva")
    print(f"==================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
