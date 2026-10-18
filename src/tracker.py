from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


class SimpleTracklet:
    """Internal track representation with position, velocity, and history."""
    _next_id = 1

    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float):
        self.track_id: int = SimpleTracklet._next_id
        SimpleTracklet._next_id += 1
        self.bbox: Tuple[int, int, int, int] = bbox
        self.confidence: float = confidence
        self.hits: int = 1
        self.time_since_update: int = 0
        self.velocity: np.ndarray = np.zeros(4, dtype=np.float32)

    def predict(self) -> Tuple[int, int, int, int]:
        """Predicts next bounding box using estimated linear velocity."""
        curr = np.array(self.bbox, dtype=np.float32)
        pred = curr + self.velocity
        self.bbox = (int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3]))
        self.time_since_update += 1
        return self.bbox

    def update(self, new_bbox: Tuple[int, int, int, int], confidence: float) -> None:
        """Updates tracklet state with a newly associated detection."""
        old_bbox = np.array(self.bbox, dtype=np.float32)
        new_arr = np.array(new_bbox, dtype=np.float32)
        # Smooth velocity estimate
        self.velocity = 0.6 * (new_arr - old_bbox) + 0.4 * self.velocity
        self.bbox = new_bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0


class MultiFaceTracker:
    """
    ByteTrack / IoU Multi-Object Tracker for faces across frames.
    Handles track association, new track creation, and lost track timeouts.
    """
    def __init__(self, iou_threshold: float = 0.35, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[SimpleTracklet] = []

    def update(
        self,
        detections: List[Dict[str, Any]],
        is_detection_frame: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Updates tracks.
        detections: [{'bbox': (x1, y1, x2, y2), 'confidence': float, 'crop': np.ndarray}]
        Returns active tracks with track_id and bounding boxes.
        """
        # Step 1: Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()

        if not is_detection_frame:
            # On skipped frames, return current predicted tracks without association
            return [
                {
                    "track_id": t.track_id,
                    "bbox": t.bbox,
                    "confidence": t.confidence,
                    "crop": None
                }
                for t in self.tracks
                if t.time_since_update <= self.max_age
            ]

        # Step 2: Match detections to existing tracks via IoU
        matched_tracks = set()
        matched_dets = set()

        if self.tracks and detections:
            cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    cost_matrix[i, j] = compute_iou(t.bbox, d["bbox"])

            # Greedy bipartite matching
            while True:
                max_iou = np.max(cost_matrix)
                if max_iou < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
                if i not in matched_tracks and j not in matched_dets:
                    self.tracks[i].update(detections[j]["bbox"], detections[j]["confidence"])
                    # Attach crop to matched detection for embedding extraction
                    detections[j]["track_id"] = self.tracks[i].track_id
                    matched_tracks.add(i)
                    matched_dets.add(j)
                cost_matrix[i, :] = -1.0
                cost_matrix[:, j] = -1.0

        # Step 3: Initiate new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                new_track = SimpleTracklet(det["bbox"], det["confidence"])
                det["track_id"] = new_track.track_id
                self.tracks.append(new_track)

        # Step 4: Purge dead tracks that exceeded max_age
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Step 5: Format output
        output = []
        for det in detections:
            if "track_id" in det:
                output.append({
                    "track_id": det["track_id"],
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "crop": det.get("crop"),
                    "embedding": det.get("embedding")
                })
        return output
