import os
from enum import Enum
from typing import Dict, List, Optional, Set
from datetime import datetime
import numpy as np
import cv2

from app.storage.base import DatabaseRepository
from app.config import StorageConfig, TrackingConfig
from app.tracker import KalmanTracklet
from app.logger import get_logger


class VisitorState(str, Enum):
    OUTSIDE = "OUTSIDE"
    TENTATIVE = "TENTATIVE"
    INSIDE = "INSIDE"
    EXIT_PENDING = "EXIT_PENDING"


class VisitorSession:
    """Tracks the state machine and metadata for an active visitor encounter."""
    def __init__(self, track_id: int, frame_idx: int):
        self.track_id: int = track_id
        self.face_id: Optional[str] = None
        self.state: VisitorState = VisitorState.TENTATIVE
        self.first_frame: int = frame_idx
        self.last_frame: int = frame_idx
        self.entry_logged: bool = False
        self.exit_logged: bool = False
        self.entry_time: Optional[str] = None
        self.exit_time: Optional[str] = None
        self.entry_image_path: Optional[str] = None
        self.exit_image_path: Optional[str] = None
        self.best_crop: Optional[np.ndarray] = None


class EventManager:
    """
    State machine enforcing strictly ONE Entry and ONE Exit event per physical visit.
    Guarantees:
    1. Atomic filesystem image writes (.tmp -> os.replace) to eliminate corrupted crops.
    2. Zero duplicate events via per-identity state tracking (OUTSIDE -> INSIDE -> OUTSIDE).
    3. Flushes exit events for any remaining in-frame visitors upon stream completion or shutdown.
    """
    def __init__(
        self,
        repo: DatabaseRepository,
        storage_cfg: StorageConfig,
        tracking_cfg: TrackingConfig
    ):
        self.repo = repo
        self.storage_cfg = storage_cfg
        self.tracking_cfg = tracking_cfg
        self.logger = get_logger("EventManager")

        # Active sessions keyed by track_id: {track_id: VisitorSession}
        self.active_sessions: Dict[int, VisitorSession] = {}
        # Set of face_ids currently INSIDE the frame to strictly prevent duplicate entries
        self.inside_face_ids: Set[str] = set()

    def _atomic_save_crop(
        self,
        crop: Optional[np.ndarray],
        face_id: str,
        event_type: str,
        timestamp_str: str
    ) -> str:
        """
        Atomically saves a cropped face image to:
        logs/{entries|exits}/YYYY-MM-DD/{face_id}_{event_type}_{safe_ts}.jpg
        using write-to-temp and atomic rename (os.replace).
        """
        date_folder = datetime.now().strftime("%Y-%m-%d")
        base_dir = self.storage_cfg.entries_dir if event_type.lower() == "entry" else self.storage_cfg.exits_dir
        target_dir = os.path.join(base_dir, date_folder)
        os.makedirs(target_dir, exist_ok=True)

        safe_ts = timestamp_str.replace(":", "-").replace(".", "_")
        filename = f"{face_id}_{event_type.lower()}_{safe_ts}.jpg"
        final_path = os.path.join(target_dir, filename)
        temp_path = os.path.join(target_dir, f".tmp_{filename}")

        if crop is not None and crop.size > 0:
            try:
                # 1. Write to temporary file
                cv2.imwrite(temp_path, crop)
                # 2. Atomic filesystem replace
                os.replace(temp_path, final_path)
            except Exception as e:
                self.logger.error(f"Error during atomic image write: {e}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                return ""
        return final_path

    def process_tracks(
        self,
        active_tracks: List[KalmanTracklet],
        frame_idx: int,
        timestamp_str: str
    ) -> None:
        """
        Evaluates state transitions for all active tracks on the current frame:
        - Promotes TENTATIVE -> INSIDE and logs ENTRY once confirmed and identity is bound.
        - Promotes INSIDE -> OUTSIDE and logs EXIT when track expires.
        """
        current_track_ids = set()

        for track in active_tracks:
            t_id = track.track_id
            current_track_ids.add(t_id)

            if t_id not in self.active_sessions:
                self.active_sessions[t_id] = VisitorSession(t_id, frame_idx)

            session = self.active_sessions[t_id]
            session.last_frame = frame_idx
            if track.face_id is not None:
                session.face_id = track.face_id

            if track.best_crop is not None:
                session.best_crop = track.best_crop

            # Evaluate ENTRY Trigger
            # Condition: Hits >= min_confirm_frames, face_id is bound, and ENTRY not yet logged
            if (
                track.hits >= self.tracking_cfg.min_confirm_frames
                and session.face_id is not None
                and not session.entry_logged
                and session.face_id not in self.inside_face_ids
            ):
                session.state = VisitorState.INSIDE
                session.entry_logged = True
                session.entry_time = timestamp_str
                self.inside_face_ids.add(session.face_id)

                # 1. Save atomic entry crop
                crop_path = self._atomic_save_crop(session.best_crop, session.face_id, "entry", timestamp_str)
                session.entry_image_path = crop_path

                # 2. Insert into database events table
                self.repo.log_event(
                    face_id=session.face_id,
                    event_type="entry",
                    timestamp=timestamp_str,
                    image_path=crop_path,
                    track_id=t_id
                )

                # 3. Write structured audit log
                self.logger.info(
                    f"[ENTRY] Visitor {session.face_id} (Track #{t_id}) entered frame at frame #{frame_idx}. "
                    f"Snapshot: {crop_path}"
                )

        # Evaluate EXIT Triggers
        # Identifies sessions whose tracks were dropped from active_tracks
        expired_track_ids = []
        for t_id, session in list(self.active_sessions.items()):
            if t_id not in current_track_ids:
                # Track has officially expired from tracker (patience exceeded)
                if session.entry_logged and not session.exit_logged and session.face_id is not None:
                    session.state = VisitorState.OUTSIDE
                    session.exit_logged = True
                    session.exit_time = timestamp_str
                    self.inside_face_ids.discard(session.face_id)

                    # 1. Save atomic exit crop
                    exit_crop_path = self._atomic_save_crop(session.best_crop, session.face_id, "exit", timestamp_str)
                    session.exit_image_path = exit_crop_path

                    # 2. Insert into database events table
                    self.repo.log_event(
                        face_id=session.face_id,
                        event_type="exit",
                        timestamp=timestamp_str,
                        image_path=exit_crop_path,
                        track_id=t_id
                    )

                    # 3. Write structured audit log
                    self.logger.info(
                        f"[EXIT] Visitor {session.face_id} (Track #{t_id}) exited frame after "
                        f"{session.last_frame - session.first_frame + 1} frames. Snapshot: {exit_crop_path}"
                    )

                expired_track_ids.append(t_id)

        # Clean up expired sessions from memory
        for t_id in expired_track_ids:
            del self.active_sessions[t_id]

    def flush_active_tracks(self, final_frame_idx: int, timestamp_str: str) -> None:
        """Flushes final EXIT events for any remaining in-frame tracks on stream completion or shutdown."""
        self.logger.info(f"Flushing {len(self.active_sessions)} remaining active sessions at stream termination...")
        for t_id, session in list(self.active_sessions.items()):
            if session.entry_logged and not session.exit_logged and session.face_id is not None:
                session.state = VisitorState.OUTSIDE
                session.exit_logged = True
                session.exit_time = timestamp_str
                self.inside_face_ids.discard(session.face_id)

                exit_crop_path = self._atomic_save_crop(session.best_crop, session.face_id, "exit", timestamp_str)
                session.exit_image_path = exit_crop_path

                self.repo.log_event(
                    face_id=session.face_id,
                    event_type="exit",
                    timestamp=timestamp_str,
                    image_path=exit_crop_path,
                    track_id=t_id
                )
                self.logger.info(
                    f"[EXIT - SHUTDOWN FLUSH] Visitor {session.face_id} (Track #{t_id}) finalized on shutdown. Snapshot: {exit_crop_path}"
                )

        self.active_sessions.clear()
        self.inside_face_ids.clear()
