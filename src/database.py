import os
import sqlite3
import logging
import json
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class SystemLogger:
    """Configures structured file logging for events.log and console output."""

    @staticmethod
    def setup_logger(log_file_path: str = "logs/events.log") -> logging.Logger:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        logger = logging.getLogger("FaceTracker")
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers if setup is called multiple times
        if not logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_format = logging.Formatter(
                "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_format)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)

            # Console handler with clean formatting
            console_handler = logging.StreamHandler()
            console_format = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S"
            )
            console_handler.setFormatter(console_format)
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)

        return logger


class DatabaseManager:
    """Manages SQLite storage for visitors, facial embeddings, and entry/exit events."""

    def __init__(self, db_path: str = "logs/visitors.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Visitors table: stores unique identities and representative embedding
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS visitors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id TEXT UNIQUE NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    total_visits INTEGER DEFAULT 1,
                    first_crop_path TEXT,
                    embedding_json TEXT
                )
            """)

            # Events table: stores every entry and exit event
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    frame_number INTEGER,
                    image_path TEXT,
                    confidence REAL,
                    FOREIGN KEY(visitor_id) REFERENCES visitors(visitor_id)
                )
            """)
            conn.commit()

    def register_new_visitor(
        self,
        visitor_id: str,
        embedding: np.ndarray,
        timestamp: str,
        crop_path: Optional[str] = None
    ) -> bool:
        """Registers a newly detected face into the database."""
        emb_json = json.dumps(embedding.tolist()) if embedding is not None else None
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO visitors (visitor_id, first_seen, last_seen, total_visits, first_crop_path, embedding_json)
                    VALUES (?, ?, ?, 1, ?, ?)
                """, (visitor_id, timestamp, timestamp, crop_path, emb_json))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def update_visitor_last_seen(self, visitor_id: str, timestamp: str) -> None:
        """Updates last seen timestamp for an existing visitor."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE visitors 
                SET last_seen = ? 
                WHERE visitor_id = ?
            """, (timestamp, visitor_id))
            conn.commit()

    def log_event(
        self,
        visitor_id: str,
        event_type: str,
        timestamp: str,
        frame_number: int,
        image_path: Optional[str] = None,
        confidence: float = 1.0
    ) -> None:
        """Logs an ENTRY or EXIT event to the events table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (visitor_id, event_type, timestamp, frame_number, image_path, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (visitor_id, event_type.upper(), timestamp, frame_number, image_path, confidence))
            conn.commit()

    def get_all_registered_visitors(self) -> List[Dict[str, Any]]:
        """Retrieves all registered visitors and their embeddings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT visitor_id, first_seen, last_seen, total_visits, first_crop_path, embedding_json FROM visitors")
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                if item["embedding_json"]:
                    item["embedding"] = np.array(json.loads(item["embedding_json"]), dtype=np.float32)
                else:
                    item["embedding"] = None
                results.append(item)
            return results

    def get_unique_visitor_count(self) -> int:
        """Returns total count of registered unique visitors."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM visitors")
            return cursor.fetchone()[0]

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent events."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT visitor_id, event_type, timestamp, frame_number, image_path, confidence 
                FROM events 
                ORDER BY id DESC 
                LIMIT ?
            """, (limit,))
            return [dict(r) for r in cursor.fetchall()]
