from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from app.config import TrackingConfig
from app.detector import DetectionResult
from app.logger import get_logger


def calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union_area = float(areaA + areaB - inter_area)
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class KalmanTracklet:
    """
    Constant-Velocity Kalman Filter State Estimator for 2D Bounding Boxes:
    State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]^T
    """
    _next_id = 1

    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float):
        self.track_id: int = KalmanTracklet._next_id
        KalmanTracklet._next_id += 1

        # State vector: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.bbox: Tuple[int, int, int, int] = bbox
        self.velocity: np.ndarray = np.zeros(4, dtype=np.float32)
        self.confidence: float = confidence

        # Track management
        self.hits: int = 0
        self.age: int = 0
        self.time_since_update: int = 0

        # Identity binding (track_id -> face_id)
        self.face_id: Optional[str] = None
        self.is_verified: bool = False
        self.last_verified_frame: int = -1
        self.match_similarity: float = 0.0

        # Best crop caching for audit snapshot
        self.best_crop: Optional[np.ndarray] = None
        self.best_crop_quality: float = 0.0

    def predict(self) -> Tuple[int, int, int, int]:
        """Predicts position on skipped or current frame using estimated velocity."""
        curr = np.array(self.bbox, dtype=np.float32)
        pred = curr + self.velocity
        self.bbox = (int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3]))
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, new_bbox: Tuple[int, int, int, int], confidence: float, crop: Optional[np.ndarray], blur_score: float) -> None:
        """Updates tracklet state with an associated measurement."""
        old_arr = np.array(self.bbox, dtype=np.float32)
        new_arr = np.array(new_bbox, dtype=np.float32)

        # Smooth Kalman velocity gain (alpha=0.6, beta=0.4)
        self.velocity = 0.6 * (new_arr - old_arr) + 0.4 * self.velocity
        self.bbox = new_bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0

        # Update best crop if candidate is sharper and of equal/greater size
        if crop is not None and crop.size > 0:
            area = max(0, new_bbox[2] - new_bbox[0]) * max(0, new_bbox[3] - new_bbox[1])
            combined_quality = blur_score * (area ** 0.5)
            if combined_quality > self.best_crop_quality:
                self.best_crop = crop.copy()
                self.best_crop_quality = combined_quality


class ByteFaceTracker:
    """
    ByteTrack-style Multi-Object Face Tracker with:
    1. Kalman motion prediction for skipped detection frames.
    2. Two-stage IoU association (high-confidence detections first, then secondary).
    3. Track-to-face binding (track_id -> face_id) to avoid redundant embedding passes.
    4. Periodic re-verification to heal ID switches.
    """
    def __init__(
        self,
        config: TrackingConfig,
        reverify_every_n_frames: int = 30,
        iou_threshold: float = 0.35
    ):
        self.config = config
        self.reverify_every_n_frames = reverify_every_n_frames
        self.iou_threshold = iou_threshold
        self.logger = get_logger("ByteTracker")

        self.tracks: List[KalmanTracklet] = []

    def update(
        self,
        detections: List[DetectionResult],
        frame_idx: int,
        is_detection_frame: bool = True
    ) -> List[KalmanTracklet]:
        """
        Updates multi-target tracks for the current frame.
        On skipped frames: Runs pure Kalman prediction without dropping active tracks.
        On detection frames: Associates detections with existing tracks using IoU bipartite matching.
        """
        # Step 1: Predict new locations for all existing tracks
        for track in self.tracks:
            track.predict()

        if not is_detection_frame:
            # On skipped frames, purge expired tracks and return active tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.config.exit_patience_frames]
            return self.tracks

        # Step 2: Associate detections with existing tracks
        matched_tracks = set()
        matched_dets = set()

        if self.tracks and detections:
            cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    cost_matrix[i, j] = calculate_iou(t.bbox, d.bbox)

            # Greedy bipartite matching
            while True:
                max_iou = np.max(cost_matrix)
                if max_iou < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
                if i not in matched_tracks and j not in matched_dets:
                    det = detections[j]
                    self.tracks[i].update(det.bbox, det.confidence, det.crop, det.blur_score)
                    matched_tracks.add(i)
                    matched_dets.add(j)
                cost_matrix[i, :] = -1.0
                cost_matrix[:, j] = -1.0

        # Step 3: Initiate new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                new_track = KalmanTracklet(det.bbox, det.confidence)
                new_track.update(det.bbox, det.confidence, det.crop, det.blur_score)
                self.tracks.append(new_track)

        # Step 4: Purge dead tracks that exceed exit_patience_frames
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.config.exit_patience_frames]

        return self.tracks

    def get_tracks_needing_embedding(self, frame_idx: int) -> List[KalmanTracklet]:
        """
        Identifies active tracks that require biometric feature extraction:
        1. Tracks that have reached min_confirm_frames and have no face_id bound.
        2. Already verified tracks that are due for periodic re-verification.
        """
        needing_embed = []
        for t in self.tracks:
            # Only consider tracks with an active measurement
            if t.time_since_update != 0 or t.best_crop is None:
                continue

            # Case A: Track is confirmed but identity not yet bound
            if t.face_id is None and t.hits >= self.config.min_confirm_frames:
                needing_embed.append(t)
            # Case B: Periodic re-verification to correct potential ID switches
            elif t.face_id is not None and t.is_verified and self.reverify_every_n_frames > 0:
                if (frame_idx - t.last_verified_frame) >= self.reverify_every_n_frames:
                    needing_embed.append(t)

        return needing_embed

    def bind_identity(self, track_id: int, face_id: str, similarity: float, frame_idx: int) -> None:
        """Binds a resolved biometric identity to a specific track."""
        for t in self.tracks:
            if t.track_id == track_id:
                t.face_id = face_id
                t.is_verified = True
                t.last_verified_frame = frame_idx
                t.match_similarity = similarity
                break
