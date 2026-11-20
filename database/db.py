"""
database/db.py
SQLite Database Manager for AI-Based Automatic License Plate Recognition (ANPR) System.
Handles schema initialization, CRUD operations, analytics aggregation, and CSV export.
"""

import os
import sqlite3
import csv
import io
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

DB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DB_DIR)
DB_PATH = os.path.join(DB_DIR, "anpr_system.db")


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite database with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schema and indexes."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number TEXT NOT NULL,
        raw_text TEXT,
        vehicle_type TEXT DEFAULT 'Car',
        confidence REAL DEFAULT 0.0,
        vehicle_confidence REAL DEFAULT 0.0,
        state_code TEXT,
        state_name TEXT,
        is_valid INTEGER DEFAULT 1,
        format_type TEXT DEFAULT 'Indian Standard (HSRP)',
        image_path TEXT,
        crop_path TEXT,
        source_type TEXT DEFAULT 'image',
        source_name TEXT DEFAULT '',
        timestamp TEXT NOT NULL,
        detection_date TEXT NOT NULL,
        detection_time TEXT NOT NULL
    );
    """)

    # Create helpful indexes for fast search and analytics
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_date ON detections(detection_date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_type ON detections(vehicle_type);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plate_number ON detections(plate_number);")

    conn.commit()

    # Prepopulate with initial historical samples if database is brand new
    cursor.execute("SELECT COUNT(*) FROM detections")
    count = cursor.fetchone()[0]
    if count == 0:
        _seed_sample_data(cursor)
        conn.commit()

    conn.close()


def _seed_sample_data(cursor: sqlite3.Cursor):
    """Seeds realistic historical records for viva demonstration & analytics showcase."""
    now = datetime.now()
    sample_records = [
        ("DL01AB1234", "DL 01 AB 1234", "Car", 0.94, 0.91, "DL", "Delhi", 1, "Indian Standard (HSRP)", "results/images/sample_1.jpg", "results/crops/crop_DL01AB1234.png", "image", "sample_1.jpg", (now - timedelta(hours=1)).isoformat(), (now - timedelta(hours=1)).strftime("%Y-%m-%d"), (now - timedelta(hours=1)).strftime("%H:%M:%S")),
        ("MH12DE1433", "MH 12 DE 1433", "SUV", 0.96, 0.89, "MH", "Maharashtra", 1, "Indian Standard (HSRP)", "results/images/sample_2.jpg", "results/crops/crop_MH12DE1433.png", "image", "sample_2.jpg", (now - timedelta(hours=3)).isoformat(), (now - timedelta(hours=3)).strftime("%Y-%m-%d"), (now - timedelta(hours=3)).strftime("%H:%M:%S")),
        ("KA03MG2341", "KA 03 MG 2341", "Motorcycle", 0.88, 0.85, "KA", "Karnataka", 1, "Indian Standard (HSRP)", "results/images/sample_3.jpg", "results/crops/crop_KA03MG2341.png", "image", "sample_3.jpg", (now - timedelta(hours=5)).isoformat(), (now - timedelta(hours=5)).strftime("%Y-%m-%d"), (now - timedelta(hours=5)).strftime("%H:%M:%S")),
        ("HR26DK8337", "HR 26 DK 8337", "Car", 0.95, 0.92, "HR", "Haryana", 1, "Indian Standard (HSRP)", "results/images/sample_4.jpg", "results/crops/crop_HR26DK8337.png", "image", "sample_4.jpg", (now - timedelta(hours=7)).isoformat(), (now - timedelta(hours=7)).strftime("%Y-%m-%d"), (now - timedelta(hours=7)).strftime("%H:%M:%S")),
        ("22BH1234AA", "22 BH 1234 AA", "Car", 0.92, 0.88, "BH", "Bharat Series (Pan-India)", 1, "Bharat Series (BH)", "results/images/sample_5.jpg", "results/crops/crop_22BH1234AA.png", "image", "sample_5.jpg", (now - timedelta(days=1, hours=2)).isoformat(), (now - timedelta(days=1)).strftime("%Y-%m-%d"), "10:14:22"),
        ("UP32BN4590", "UP 32 BN 4590", "Bus", 0.91, 0.94, "UP", "Uttar Pradesh", 1, "Indian Standard (HSRP)", "", "", "video", "traffic_cam_01.mp4", (now - timedelta(days=1, hours=4)).isoformat(), (now - timedelta(days=1)).strftime("%Y-%m-%d"), "12:35:10"),
        ("TN09AZ7821", "TN 09 AZ 7821", "Truck", 0.89, 0.87, "TN", "Tamil Nadu", 1, "Indian Standard (HSRP)", "", "", "video", "highway_toll.mp4", (now - timedelta(days=2, hours=3)).isoformat(), (now - timedelta(days=2)).strftime("%Y-%m-%d"), "15:20:45"),
        ("GJ01KJ9988", "GJ 01 KJ 9988", "Car", 0.97, 0.93, "GJ", "Gujarat", 1, "Indian Standard (HSRP)", "", "", "image", "tollgate_feed.jpg", (now - timedelta(days=3)).isoformat(), (now - timedelta(days=3)).strftime("%Y-%m-%d"), "09:40:12"),
        ("RJ14CD5521", "RJ 14 CD 5521", "Motorcycle", 0.86, 0.82, "RJ", "Rajasthan", 1, "Indian Standard (HSRP)", "", "", "image", "parking_gate.jpg", (now - timedelta(days=4)).isoformat(), (now - timedelta(days=4)).strftime("%Y-%m-%d"), "18:05:33"),
        ("WB02AC3412", "WB 02 AC 3412", "Car", 0.93, 0.90, "WB", "West Bengal", 1, "Indian Standard (HSRP)", "", "", "image", "security_gate.jpg", (now - timedelta(days=5)).isoformat(), (now - timedelta(days=5)).strftime("%Y-%m-%d"), "11:15:20")
    ]

    cursor.executemany("""
    INSERT INTO detections (
        plate_number, raw_text, vehicle_type, confidence, vehicle_confidence,
        state_code, state_name, is_valid, format_type, image_path, crop_path,
        source_type, source_name, timestamp, detection_date, detection_time
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_records)


def insert_detection(data: Dict[str, Any]) -> int:
    """Inserts a new detection record into the database and returns the generated row id."""
    conn = get_db_connection()
    cursor = conn.cursor()

    now = datetime.now()
    timestamp = data.get("timestamp", now.isoformat())
    detection_date = data.get("detection_date", now.strftime("%Y-%m-%d"))
    detection_time = data.get("detection_time", now.strftime("%H:%M:%S"))

    cursor.execute("""
    INSERT INTO detections (
        plate_number, raw_text, vehicle_type, confidence, vehicle_confidence,
        state_code, state_name, is_valid, format_type, image_path, crop_path,
        source_type, source_name, timestamp, detection_date, detection_time
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("plate_number", "UNKNOWN"),
        data.get("raw_text", ""),
        data.get("vehicle_type", "Car"),
        float(data.get("confidence", 0.0)),
        float(data.get("vehicle_confidence", 0.0)),
        data.get("state_code", "UNKNOWN"),
        data.get("state_name", "Unknown State"),
        1 if data.get("is_valid", True) else 0,
        data.get("format_type", "Standard"),
        data.get("image_path", ""),
        data.get("crop_path", ""),
        data.get("source_type", "image"),
        data.get("source_name", ""),
        timestamp,
        detection_date,
        detection_time
    ))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_recent_detections(limit: int = 10) -> List[Dict[str, Any]]:
    """Fetches the most recent detections."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_detections(
    search: str = "",
    vehicle_type: str = "",
    date_from: str = "",
    date_to: str = "",
    is_valid: Optional[int] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Queries detections with filtering, search, and pagination."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM detections WHERE 1=1"
    params: List[Any] = []

    if search:
        query += " AND (plate_number LIKE ? OR state_name LIKE ? OR state_code LIKE ?)"
        s = f"%{search.strip().upper()}%"
        params.extend([s, s, s])

    if vehicle_type and vehicle_type.lower() != "all":
        query += " AND LOWER(vehicle_type) = LOWER(?)"
        params.append(vehicle_type.strip())

    if date_from:
        query += " AND detection_date >= ?"
        params.append(date_from)

    if date_to:
        query += " AND detection_date <= ?"
        params.append(date_to)

    if is_valid is not None:
        query += " AND is_valid = ?"
        params.append(is_valid)

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def count_detections(
    search: str = "",
    vehicle_type: str = "",
    date_from: str = "",
    date_to: str = "",
    is_valid: Optional[int] = None
) -> int:
    """Counts matching detections for pagination."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT COUNT(*) FROM detections WHERE 1=1"
    params: List[Any] = []

    if search:
        query += " AND (plate_number LIKE ? OR state_name LIKE ? OR state_code LIKE ?)"
        s = f"%{search.strip().upper()}%"
        params.extend([s, s, s])

    if vehicle_type and vehicle_type.lower() != "all":
        query += " AND LOWER(vehicle_type) = LOWER(?)"
        params.append(vehicle_type.strip())

    if date_from:
        query += " AND detection_date >= ?"
        params.append(date_from)

    if date_to:
        query += " AND detection_date <= ?"
        params.append(date_to)

    if is_valid is not None:
        query += " AND is_valid = ?"
        params.append(is_valid)

    cursor.execute(query, params)
    total = cursor.fetchone()[0]
    conn.close()
    return total


def delete_detection(record_id: int) -> bool:
    """Deletes a detection record by its primary key ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def clear_all_detections() -> bool:
    """Clears all detection records from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    conn.commit()
    conn.close()
    return True


def get_statistics() -> Dict[str, Any]:
    """Computes high-level analytics KPIs for the dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total vehicles & plates detected
    cursor.execute("SELECT COUNT(*) FROM detections")
    total_detections = cursor.fetchone()[0]

    # Today's detections
    today_str = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM detections WHERE detection_date = ?", (today_str,))
    today_detections = cursor.fetchone()[0]

    # Average confidence
    cursor.execute("SELECT AVG(confidence) FROM detections WHERE confidence > 0")
    avg_conf_row = cursor.fetchone()[0]
    avg_confidence = round(float(avg_conf_row * 100), 1) if avg_conf_row else 0.0

    # Valid plates count
    cursor.execute("SELECT COUNT(*) FROM detections WHERE is_valid = 1")
    valid_count = cursor.fetchone()[0]
    valid_rate = round((valid_count / total_detections * 100), 1) if total_detections > 0 else 0.0

    # Total unique plates
    cursor.execute("SELECT COUNT(DISTINCT plate_number) FROM detections")
    unique_plates = cursor.fetchone()[0]

    conn.close()

    return {
        "total_detections": total_detections,
        "today_detections": today_detections,
        "avg_confidence": avg_confidence,
        "valid_count": valid_count,
        "valid_rate": valid_rate,
        "unique_plates": unique_plates
    }


def get_vehicle_distribution() -> Dict[str, int]:
    """Returns vehicle distribution count by category for Chart.js doughnut chart."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT vehicle_type, COUNT(*) as count 
        FROM detections 
        GROUP BY vehicle_type 
        ORDER BY count DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    dist = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0, "SUV": 0, "Other": 0}
    for row in rows:
        v_type = row["vehicle_type"].capitalize()
        if v_type in dist:
            dist[v_type] += row["count"]
        elif "bike" in v_type.lower() or "motor" in v_type.lower():
            dist["Motorcycle"] += row["count"]
        else:
            dist["Other"] += row["count"]
    return dist


def get_daily_analytics(days: int = 7) -> List[Dict[str, Any]]:
    """Returns daily detection frequency for the last N days for Chart.js line/bar chart."""
    conn = get_db_connection()
    cursor = conn.cursor()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)

    cursor.execute("""
        SELECT detection_date, COUNT(*) as count 
        FROM detections 
        WHERE detection_date >= ? 
        GROUP BY detection_date 
        ORDER BY detection_date ASC
    """, (start_date.strftime("%Y-%m-%d"),))

    results = {row["detection_date"]: row["count"] for row in cursor.fetchall()}
    conn.close()

    daily_data = []
    for i in range(days):
        d = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        daily_data.append({
            "date": d,
            "label": (start_date + timedelta(days=i)).strftime("%b %d"),
            "count": results.get(d, 0)
        })

    return daily_data


def get_state_distribution(limit: int = 5) -> List[Dict[str, Any]]:
    """Returns top detected states for Chart.js horizontal bar chart."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT state_name, COUNT(*) as count 
        FROM detections 
        WHERE state_name IS NOT NULL AND state_name != 'Unknown State'
        GROUP BY state_name 
        ORDER BY count DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [{"state": row["state_name"], "count": row["count"]} for row in rows]


def export_to_csv_string() -> str:
    """Exports all detection records into a CSV formatted string."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, plate_number, vehicle_type, confidence, vehicle_confidence,
               state_code, state_name, format_type, is_valid, source_type,
               timestamp, detection_date, detection_time
        FROM detections ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Plate Number", "Vehicle Type", "OCR Confidence", "Vehicle Confidence",
        "State Code", "State Name", "Format Type", "Is Valid", "Source Type",
        "Timestamp", "Date", "Time"
    ])

    for r in rows:
        writer.writerow([
            r["id"], r["plate_number"], r["vehicle_type"], f"{r['confidence']:.2f}",
            f"{r['vehicle_confidence']:.2f}", r["state_code"], r["state_name"],
            r["format_type"], "Valid" if r["is_valid"] else "Invalid",
            r["source_type"], r["timestamp"], r["detection_date"], r["detection_time"]
        ])

    return output.getvalue()
