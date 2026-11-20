"""
src/database.py
SQLite Database Handler for License Plate Records, Security Rules, and Analytics.
"""

import sqlite3
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "database", "alpr_database.db")


class ALPRDatabase:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes tables for plate detection logs and security whitelist/blacklist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Detections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL,
                    raw_text TEXT,
                    confidence REAL NOT NULL,
                    state_code TEXT,
                    state_name TEXT,
                    vehicle_type TEXT DEFAULT 'Car',
                    source TEXT DEFAULT 'Image',
                    crop_image_path TEXT,
                    full_image_path TEXT,
                    status TEXT DEFAULT 'Allowed',
                    notes TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Blacklist / Whitelist table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT UNIQUE NOT NULL,
                    owner_name TEXT NOT NULL,
                    category TEXT NOT NULL CHECK(category IN ('Whitelisted', 'Blacklisted', 'VIP')),
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Add default security rules if empty
            cursor.execute("SELECT COUNT(*) FROM security_rules")
            if cursor.fetchone()[0] == 0:
                default_rules = [
                    ("DL01AB1234", "Dr. R. Sharma (Director)", "Whitelisted", "Staff VIP Parking Bay A-1"),
                    ("MH12DE1433", "Prof. S. Verma (HOD AI/ML)", "Whitelisted", "Faculty Parking Bay B-4"),
                    ("HR26DK8337", "Unknown (Flagged Suspect)", "Blacklisted", "SECURITY ALERT: Reported stolen vehicle"),
                    ("KA05MB9999", "Student Council Van", "Whitelisted", "Campus delivery permit"),
                    ("UP16BV0007", "Ex-Employee", "Blacklisted", "Entry revoked - No campus access")
                ]
                cursor.executemany("""
                    INSERT OR IGNORE INTO security_rules (plate_number, owner_name, category, notes)
                    VALUES (?, ?, ?, ?)
                """, default_rules)

            # Users table for secure authentication
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] == 0:
                def _hash(p: str) -> str:
                    return hashlib.sha256(p.encode('utf-8')).hexdigest()

                default_users = [
                    ("admin", _hash("admin123"), "System Administrator (Faculty)", "Administrator"),
                    ("guard", _hash("guard123"), "Campus Security Officer", "Security Officer"),
                    ("evaluator", _hash("viva2026"), "External B.Tech Examiner", "Examiner / Evaluator"),
                    ("student", _hash("student123"), "AI/ML Project Student", "Examiner / Evaluator")
                ]
                cursor.executemany("""
                    INSERT OR IGNORE INTO users (username, password_hash, full_name, role)
                    VALUES (?, ?, ?, ?)
                """, default_users)

            conn.commit()

    def check_security_status(self, plate_number: str) -> Dict[str, Any]:
        """Checks if a plate is whitelisted, blacklisted, or visitor."""
        clean_num = plate_number.replace(" ", "").upper()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT category, owner_name, notes FROM security_rules WHERE plate_number = ?", (clean_num,))
            row = cursor.fetchone()
            if row:
                cat = row["category"]
                status = "Flagged" if cat == "Blacklisted" else "Whitelisted"
                return {
                    "category": cat,
                    "status": status,
                    "owner": row["owner_name"],
                    "notes": row["notes"],
                    "is_alert": (cat == "Blacklisted")
                }
            return {
                "category": "Visitor",
                "status": "Visitor / Allowed",
                "owner": "Guest Vehicle",
                "notes": "Standard visitor entry",
                "is_alert": False
            }

    def add_detection(
        self,
        plate_number: str,
        raw_text: str,
        confidence: float,
        state_code: str = "N/A",
        state_name: str = "Unknown",
        vehicle_type: str = "Car",
        source: str = "Image",
        crop_image_path: str = "",
        full_image_path: str = "",
        notes: str = ""
    ) -> int:
        """Logs a detected plate into the database, resolving its security status."""
        sec_info = self.check_security_status(plate_number)
        status = sec_info["status"]
        if sec_info["is_alert"]:
            notes = f"SECURITY ALERT: {sec_info['notes']}" if not notes else f"{notes} | {sec_info['notes']}"
        elif sec_info["category"] == "Whitelisted":
            notes = f"Whitelisted: {sec_info['owner']} ({sec_info['notes']})"

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detections (
                    plate_number, raw_text, confidence, state_code, state_name,
                    vehicle_type, source, crop_image_path, full_image_path,
                    status, notes, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plate_number.replace(" ", "").upper(),
                raw_text,
                round(float(confidence), 2),
                state_code,
                state_name,
                vehicle_type,
                source,
                crop_image_path,
                full_image_path,
                status,
                notes,
                now
            ))
            conn.commit()
            return cursor.lastrowid

    def get_detections(
        self,
        limit: int = 100,
        search_query: Optional[str] = None,
        source_filter: Optional[str] = None,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves detection logs with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM detections WHERE 1=1"
            params = []

            if search_query:
                query += " AND (plate_number LIKE ? OR state_name LIKE ? OR notes LIKE ?)"
                q = f"%{search_query.strip()}%"
                params.extend([q, q, q])

            if source_filter and source_filter != "All":
                query += " AND source = ?"
                params.append(source_filter)

            if status_filter and status_filter != "All":
                query += " AND status LIKE ?"
                params.append(f"%{status_filter}%")

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_dataframe(self) -> pd.DataFrame:
        """Returns all detections as a pandas DataFrame."""
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC", conn)
            return df

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates project analytics for presentation and dashboard widgets."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total detections
            cursor.execute("SELECT COUNT(*) FROM detections")
            total_detections = cursor.fetchone()[0]

            # Unique plates
            cursor.execute("SELECT COUNT(DISTINCT plate_number) FROM detections")
            unique_plates = cursor.fetchone()[0]

            # Flagged / Blacklisted count
            cursor.execute("SELECT COUNT(*) FROM detections WHERE status LIKE '%Flagged%'")
            flagged_count = cursor.fetchone()[0]

            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM detections")
            avg_conf_row = cursor.fetchone()[0]
            avg_confidence = round(avg_conf_row, 1) if avg_conf_row else 0.0

            # State breakdown (Top 5)
            cursor.execute("""
                SELECT state_name, COUNT(*) as cnt 
                FROM detections 
                WHERE state_name != 'Unknown' 
                GROUP BY state_name 
                ORDER BY cnt DESC 
                LIMIT 5
            """)
            top_states = [dict(row) for row in cursor.fetchall()]

            # Source breakdown
            cursor.execute("SELECT source, COUNT(*) as cnt FROM detections GROUP BY source")
            sources = [dict(row) for row in cursor.fetchall()]

            return {
                "total_detections": total_detections,
                "unique_plates": unique_plates,
                "flagged_count": flagged_count,
                "avg_confidence": avg_confidence,
                "top_states": top_states,
                "sources": sources
            }

    def delete_detection(self, detection_id: int) -> bool:
        """Deletes a single detection record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all_detections(self):
        """Clears all detection records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detections")
            conn.commit()

    # Security Rules management
    def get_security_rules(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM security_rules ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]

    def add_security_rule(self, plate_number: str, owner_name: str, category: str, notes: str = "") -> bool:
        clean_num = plate_number.replace(" ", "").upper()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO security_rules (plate_number, owner_name, category, notes)
                    VALUES (?, ?, ?, ?)
                """, (clean_num, owner_name, category, notes))
                conn.commit()
                return True
            except Exception:
                return False

    def delete_security_rule(self, rule_id: int) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM security_rules WHERE id = ?", (rule_id,))
            conn.commit()
            return cursor.rowcount > 0

    # User Authentication Management
    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Verifies username and password against users table."""
        if not username or not password:
            return None
        p_hash = self.hash_password(password)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, username, full_name, role, created_at
                FROM users
                WHERE LOWER(username) = LOWER(?) AND password_hash = ?
            """, (username.strip(), p_hash))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_users(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, full_name, role, created_at FROM users ORDER BY id ASC")
            return [dict(row) for row in cursor.fetchall()]

    def add_user(self, username: str, password: str, full_name: str, role: str) -> tuple[bool, str]:
        username_clean = username.strip().lower()
        if len(username_clean) < 3:
            return False, "Username must be at least 3 characters long."
        if len(password) < 4:
            return False, "Password must be at least 4 characters long."
        p_hash = self.hash_password(password)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE LOWER(username) = LOWER(?)", (username_clean,))
            if cursor.fetchone():
                return False, f"Username '{username_clean}' is already registered. Please choose another."
            try:
                cursor.execute("""
                    INSERT INTO users (username, password_hash, full_name, role)
                    VALUES (?, ?, ?, ?)
                """, (username_clean, p_hash, full_name.strip(), role))
                conn.commit()
                return True, "Account registered successfully! You can now sign in."
            except Exception as e:
                return False, f"Database error during registration: {e}"
