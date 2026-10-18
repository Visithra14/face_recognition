import os
import cv2
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from src.database import DatabaseManager, SystemLogger


class TrackSession:
    """Represents an active tracking session for a single person in the video feed."""
    def __init__(self, track_id: int, initial_bbox: Tuple[int, int, int, int], frame_idx: int):
        self.track_id: int = track_id
        self.bbox: Tuple[int, int, int, int] = initial_bbox  # (x1, y1, x2, y2)
        self.first_frame: int = frame_idx
        self.last_frame: int = frame_idx
        self.lost_frames: int = 0
        
        # Identity
        self.visitor_id: Optional[str] = None
        self.embedding: Optional[np.ndarray] = None
        self.is_new_registration: bool = False
        
        # State: 'TENTATIVE', 'CONFIRMED_ACTIVE', 'EXITED'
        self.state: str = "TENTATIVE"
        self.hit_count: int = 1
        self.entry_logged: bool = False
        self.exit_logged: bool = False
        
        # Best crop for snapshot logging
        self.best_crop: Optional[np.ndarray] = None
        self.best_crop_area: int = 0
        self.entry_crop_path: Optional[str] = None
        self.exit_crop_path: Optional[str] = None
        
        # Timestamps
        self.entry_time: Optional[str] = None
        self.exit_time: Optional[str] = None


class LifecycleManager:
    """
    Manages the lifecycle state machine for face tracking:
    - Debounces detections (eliminates 1-frame glitches)
    - Associates embeddings with known visitors or auto-registers new visitors
    - Guarantees EXACTLY ONE entry and ONE exit event per physical encounter
    - Saves timestamped crop images in logs/entries/YYYY-MM-DD/ and logs/exits/YYYY-MM-DD/
    """
    def __init__(
        self,
        db_manager: DatabaseManager,
        logger,
        entries_dir: str = "logs/entries",
        exits_dir: str = "logs/exits",
        confirmation_frames: int = 3,
        max_lost_frames: int = 30,
        similarity_threshold: float = 0.55
    ):
        self.db = db_manager
        self.logger = logger
        self.entries_dir = entries_dir
        self.exits_dir = exits_dir
        self.confirmation_frames = confirmation_frames
        self.max_lost_frames = max_lost_frames
        self.similarity_threshold = similarity_threshold

        # In-memory registry for fast cosine matching: {visitor_id: np.ndarray embedding}
        self.known_gallery: Dict[str, np.ndarray] = {}
        self._load_gallery_from_db()

        # Active tracks: {track_id: TrackSession}
        self.active_tracks: Dict[int, TrackSession] = {}

    def _load_gallery_from_db(self) -> None:
        """Loads previously registered visitors into memory gallery."""
        visitors = self.db.get_all_registered_visitors()
        for v in visitors:
            if v["embedding"] is not None:
                self.known_gallery[v["visitor_id"]] = v["embedding"]
        self.logger.info(f"Loaded {len(self.known_gallery)} existing visitor identities from database.")

    def _get_next_visitor_id(self) -> str:
        """Generates the next sequential visitor ID (e.g. VISITOR_001)."""
        existing_count = self.db.get_unique_visitor_count()
        next_id_num = existing_count + 1
        candidate = f"VISITOR_{next_id_num:03d}"
        # Ensure uniqueness
        while candidate in self.known_gallery:
            next_id_num += 1
            candidate = f"VISITOR_{next_id_num:03d}"
        return candidate

    def match_or_register_face(self, embedding: np.ndarray) -> Tuple[str, bool, float]:
        """
        Matches a 512-d embedding against the known gallery using cosine similarity.
        Returns: (visitor_id, is_new_registration, max_similarity)
        """
        if embedding is None:
            return self._get_next_visitor_id(), True, 0.0

        best_match_id = None
        max_similarity = -1.0

        # Normalize query embedding
        norm_query = embedding / (np.linalg.norm(embedding) + 1e-6)

        for vid, gallery_emb in self.known_gallery.items():
            norm_gallery = gallery_emb / (np.linalg.norm(gallery_emb) + 1e-6)
            sim = float(np.dot(norm_query, norm_gallery))
            if sim > max_similarity:
                max_similarity = sim
                best_match_id = vid

        if best_match_id is not None and max_similarity >= self.similarity_threshold:
            # Re-identified existing visitor
            return best_match_id, False, max_similarity
        else:
            # Auto-register new visitor
            new_id = self._get_next_visitor_id()
            self.known_gallery[new_id] = norm_query
            return new_id, True, max_similarity if max_similarity > 0 else 0.0

    def _save_crop_image(self, crop: np.ndarray, event_type: str, visitor_id: str, timestamp_str: str) -> str:
        """Saves cropped face image to structured local folder: logs/{entries|exits}/YYYY-MM-DD/"""
        date_folder = datetime.now().strftime("%Y-%m-%d")
        base_dir = self.entries_dir if event_type.lower() == "entry" else self.exits_dir
        target_dir = os.path.join(base_dir, date_folder)
        os.makedirs(target_dir, exist_ok=True)

        safe_ts = timestamp_str.replace(":", "-").replace(".", "_")
        filename = f"{visitor_id}_{safe_ts}_{event_type.lower()}.jpg"
        filepath = os.path.join(target_dir, filename)

        if crop is not None and crop.size > 0:
            cv2.imwrite(filepath, crop)
        return filepath

    def process_frame_tracks(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detected_tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Updates track state machine with current frame's detections and tracks.
        detected_tracks list of dicts:
            {'track_id': int, 'bbox': (x1, y1, x2, y2), 'embedding': Optional[np.ndarray], 'crop': Optional[np.ndarray]}
        """
        current_time_str = datetime.now().isoformat()
        matched_track_ids = set()

        for det in detected_tracks:
            t_id = det["track_id"]
            bbox = det["bbox"]
            emb = det.get("embedding")
            crop = det.get("crop")
            matched_track_ids.add(t_id)

            if t_id not in self.active_tracks:
                # Create tentative session
                session = TrackSession(t_id, bbox, frame_idx)
                self.active_tracks[t_id] = session
            else:
                session = self.active_tracks[t_id]
                session.hit_count += 1
                session.lost_frames = 0
                session.bbox = bbox
                session.last_frame = frame_idx

            # Track quality crop
            x1, y1, x2, y2 = bbox
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if crop is not None and area > session.best_crop_area:
                session.best_crop = crop.copy()
                session.best_crop_area = area

            # Try to resolve identity if not yet resolved
            if session.visitor_id is None and emb is not None:
                vid, is_new, sim = self.match_or_register_face(emb)
                session.visitor_id = vid
                session.embedding = emb
                session.is_new_registration = is_new

                if is_new:
                    self.logger.info(
                        f"[REGISTRATION] New visitor auto-registered: {vid} at frame {frame_idx} (sim={sim:.2f})"
                    )

            # Confirm active state and trigger ENTRY event once confirmed
            if (
                session.hit_count >= self.confirmation_frames
                and session.visitor_id is not None
                and not session.entry_logged
            ):
                session.state = "CONFIRMED_ACTIVE"
                session.entry_logged = True
                session.entry_time = current_time_str

                # Save entry crop
                crop_to_save = session.best_crop if session.best_crop is not None else crop
                crop_path = self._save_crop_image(crop_to_save, "entry", session.visitor_id, current_time_str)
                session.entry_crop_path = crop_path

                # Database records
                if session.is_new_registration:
                    self.db.register_new_visitor(
                        visitor_id=session.visitor_id,
                        embedding=session.embedding,
                        timestamp=current_time_str,
                        crop_path=crop_path
                    )
                else:
                    self.db.update_visitor_last_seen(session.visitor_id, current_time_str)

                # Exactly one ENTRY event log
                self.db.log_event(
                    visitor_id=session.visitor_id,
                    event_type="ENTRY",
                    timestamp=current_time_str,
                    frame_number=frame_idx,
                    image_path=crop_path,
                    confidence=1.0
                )

                self.logger.info(
                    f"[ENTRY] Visitor {session.visitor_id} (Track #{t_id}) entered frame at frame {frame_idx}. "
                    f"Snapshot saved: {crop_path}"
                )

        # Handle lost tracks & EXIT events
        lost_track_ids = []
        for t_id, session in list(self.active_tracks.items()):
            if t_id not in matched_track_ids:
                session.lost_frames += 1
                if session.lost_frames > self.max_lost_frames:
                    # Finalize exit
                    if session.entry_logged and not session.exit_logged:
                        session.exit_logged = True
                        session.exit_time = current_time_str
                        exit_crop_path = self._save_crop_image(
                            session.best_crop, "exit", session.visitor_id, current_time_str
                        )
                        session.exit_crop_path = exit_crop_path

                        # Exactly one EXIT event log
                        self.db.log_event(
                            visitor_id=session.visitor_id,
                            event_type="EXIT",
                            timestamp=current_time_str,
                            frame_number=frame_idx,
                            image_path=exit_crop_path,
                            confidence=1.0
                        )

                        self.logger.info(
                            f"[EXIT] Visitor {session.visitor_id} (Track #{t_id}) exited camera frame. "
                            f"Total tracked frames: {session.last_frame - session.first_frame + 1}. "
                            f"Snapshot saved: {exit_crop_path}"
                        )
                    lost_track_ids.append(t_id)

        # Remove finalized tracks
        for t_id in lost_track_ids:
            del self.active_tracks[t_id]

        # Return list of active sessions for visualization
        active_display_list = []
        for t_id, session in self.active_tracks.items():
            if session.visitor_id is not None:
                active_display_list.append({
                    "track_id": t_id,
                    "visitor_id": session.visitor_id,
                    "bbox": session.bbox,
                    "state": session.state,
                    "is_new": session.is_new_registration
                })
        return active_display_list

    def finalize_all_sessions(self, final_frame_idx: int) -> None:
        """Call when video ends or stream stops to gracefully log exits for any remaining tracks."""
        current_time_str = datetime.now().isoformat()
        for t_id, session in list(self.active_tracks.items()):
            if session.entry_logged and not session.exit_logged:
                session.exit_logged = True
                session.exit_time = current_time_str
                exit_crop_path = self._save_crop_image(
                    session.best_crop, "exit", session.visitor_id, current_time_str
                )
                self.db.log_event(
                    visitor_id=session.visitor_id,
                    event_type="EXIT",
                    timestamp=current_time_str,
                    frame_number=final_frame_idx,
                    image_path=exit_crop_path,
                    confidence=1.0
                )
                self.logger.info(
                    f"[EXIT - STREAM END] Visitor {session.visitor_id} exited at end of stream. "
                    f"Snapshot: {exit_crop_path}"
                )
        self.active_tracks.clear()
