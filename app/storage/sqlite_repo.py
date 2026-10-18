import os
import sqlite3
from typing import List, Optional, Dict, Any
import numpy as np

from app.storage.base import DatabaseRepository, FaceRecord, EventRecord
from app.logger import get_logger


class SQLiteRepository(DatabaseRepository):
    """
    Production-grade SQLite implementation of DatabaseRepository.
    Enforces Write-Ahead Logging (WAL) for high concurrency, ACID transactions,
    and binary BLOB embedding serialization.
    """
    def __init__(self, db_path: str = "data/faces.db"):
        self.db_path = db_path
        self.logger = get_logger("SQLiteRepo")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates connection with thread timeout and row factory."""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for non-blocking concurrent reads and fast writes
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Creates tables, foreign keys, and indexes per schema specification."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    face_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    visit_count INTEGER DEFAULT 1,
                    first_crop_path TEXT
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    FOREIGN KEY (face_id) REFERENCES faces(face_id) ON DELETE CASCADE
                );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_face_id ON events(face_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);")
            conn.commit()
            self.logger.info(f"Database initialized at {self.db_path} in WAL mode.")

    def register_face(
        self,
        face_id: str,
        embedding: np.ndarray,
        timestamp: str,
        crop_path: Optional[str] = None
    ) -> bool:
        """Encodes embedding as compact float32 binary buffer and registers face."""
        if embedding is None:
            return False

        # Fast compact binary serialization
        emb_blob = np.ascontiguousarray(embedding, dtype=np.float32).tobytes()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO faces (face_id, embedding, first_seen, last_seen, visit_count, first_crop_path)
                    VALUES (?, ?, ?, ?, 1, ?)
                """, (face_id, emb_blob, timestamp, timestamp, crop_path))
                conn.commit()
                self.logger.info(f"[DB] Face registered: {face_id}")
                return True
        except sqlite3.IntegrityError:
            self.logger.warning(f"[DB] Duplicate registration rejected for face_id: {face_id}")
            return False

    def update_face_last_seen(self, face_id: str, timestamp: str) -> None:
        """Updates last-seen timestamp upon re-identification."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE faces
                SET last_seen = ?
                WHERE face_id = ?
            """, (timestamp, face_id))
            conn.commit()

    def increment_visit_count(self, face_id: str) -> None:
        """Increments visit counter when visitor returns on a subsequent encounter."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE faces
                SET visit_count = visit_count + 1
                WHERE face_id = ?
            """, (face_id,))
            conn.commit()

    def log_event(
        self,
        face_id: str,
        event_type: str,
        timestamp: str,
        image_path: str,
        track_id: int
    ) -> None:
        """Persists an entry or exit event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (face_id, event_type, timestamp, image_path, track_id)
                VALUES (?, ?, ?, ?, ?)
            """, (face_id, event_type.lower(), timestamp, image_path, track_id))
            conn.commit()
            self.logger.info(f"[DB] Logged event: {event_type.upper()} for {face_id} (Track #{track_id})")

    def get_all_faces(self) -> List[FaceRecord]:
        """Loads all registered faces, deserializing BLOB embeddings into NumPy float32 arrays."""
        records = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT face_id, embedding, first_seen, last_seen, visit_count, first_crop_path FROM faces")
            rows = cursor.fetchall()
            for row in rows:
                raw_blob = row["embedding"]
                emb_array = np.frombuffer(raw_blob, dtype=np.float32) if raw_blob else None
                records.append(FaceRecord(
                    face_id=row["face_id"],
                    embedding=emb_array,
                    first_seen=row["first_seen"],
                    last_seen=row["last_seen"],
                    visit_count=row["visit_count"],
                    first_crop_path=row["first_crop_path"]
                ))
        return records

    def get_unique_visitor_count(self) -> int:
        """Returns the total number of distinct registered identities."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(face_id) FROM faces")
            res = cursor.fetchone()
            return int(res[0]) if res else 0

    def get_recent_events(self, limit: int = 50) -> List[EventRecord]:
        """Returns recent lifecycle events."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_id, face_id, event_type, timestamp, image_path, track_id
                FROM events
                ORDER BY event_id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [
                EventRecord(
                    event_id=row["event_id"],
                    face_id=row["face_id"],
                    event_type=row["event_type"],
                    timestamp=row["timestamp"],
                    image_path=row["image_path"],
                    track_id=row["track_id"]
                )
                for row in rows
            ]

    def close(self) -> None:
        """Executes a WAL checkpoint to ensure all changes are flushed to the main database file."""
        try:
            with self._get_connection() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception as e:
            self.logger.warning(f"Error during WAL checkpoint: {e}")
