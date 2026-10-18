import unittest
import numpy as np

from app.config import TrackingConfig
from app.detector import DetectionResult
from app.tracker import ByteFaceTracker, calculate_iou


class TestByteFaceTracker(unittest.TestCase):
    def setUp(self):
        self.track_cfg = TrackingConfig(algorithm="bytetrack", min_confirm_frames=3, exit_patience_frames=5)
        self.tracker = ByteFaceTracker(self.track_cfg, reverify_every_n_frames=10, iou_threshold=0.35)

    def test_iou_calculation(self):
        boxA = (10, 10, 50, 50)
        boxB = (10, 10, 50, 50)
        self.assertAlmostEqual(calculate_iou(boxA, boxB), 1.0)

        boxC = (100, 100, 150, 150)
        self.assertAlmostEqual(calculate_iou(boxA, boxC), 0.0)

    def test_tracking_and_skipped_frame_prediction(self):
        dummy_crop = np.zeros((40, 40, 3), dtype=np.uint8)

        # Frame 1: Detection
        det1 = [DetectionResult(bbox=(100, 100, 150, 150), confidence=0.9, crop=dummy_crop, blur_score=60.0, is_quality_pass=True)]
        tracks = self.tracker.update(det1, frame_idx=1, is_detection_frame=True)
        self.assertEqual(len(tracks), 1)
        track_id = tracks[0].track_id

        # Frame 2: Skipped detection frame (no detections, pure Kalman prediction)
        tracks_skipped = self.tracker.update([], frame_idx=2, is_detection_frame=False)
        self.assertEqual(len(tracks_skipped), 1)
        self.assertEqual(tracks_skipped[0].track_id, track_id)

        # Frame 3: Detection with slight movement
        det3 = [DetectionResult(bbox=(105, 105, 155, 155), confidence=0.9, crop=dummy_crop, blur_score=65.0, is_quality_pass=True)]
        tracks_det3 = self.tracker.update(det3, frame_idx=3, is_detection_frame=True)
        self.assertEqual(len(tracks_det3), 1)
        self.assertEqual(tracks_det3[0].track_id, track_id)
        self.assertEqual(tracks_det3[0].hits, 2)

    def test_track_to_face_binding(self):
        dummy_crop = np.zeros((40, 40, 3), dtype=np.uint8)

        # Generate hits up to min_confirm_frames (3 hits)
        for f in range(1, 4):
            det = [DetectionResult(bbox=(50, 50, 100, 100), confidence=0.9, crop=dummy_crop, blur_score=70.0, is_quality_pass=True)]
            tracks = self.tracker.update(det, frame_idx=f, is_detection_frame=True)

        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertEqual(track.hits, 3)

        # Check track needs embedding
        needing = self.tracker.get_tracks_needing_embedding(frame_idx=3)
        self.assertEqual(len(needing), 1)
        self.assertEqual(needing[0].track_id, track.track_id)

        # Bind identity
        self.tracker.bind_identity(track.track_id, face_id="VISITOR_001", similarity=0.95, frame_idx=3)
        self.assertEqual(track.face_id, "VISITOR_001")
        self.assertTrue(track.is_verified)

        # In frame 4, it should NO LONGER need embedding (cached binding)
        det4 = [DetectionResult(bbox=(52, 52, 102, 102), confidence=0.9, crop=dummy_crop, blur_score=70.0, is_quality_pass=True)]
        self.tracker.update(det4, frame_idx=4, is_detection_frame=True)
        needing_after = self.tracker.get_tracks_needing_embedding(frame_idx=4)
        self.assertEqual(len(needing_after), 0)

    def test_track_expiration_after_patience_frames(self):
        dummy_crop = np.zeros((40, 40, 3), dtype=np.uint8)
        det = [DetectionResult(bbox=(10, 10, 50, 50), confidence=0.9, crop=dummy_crop, blur_score=50.0, is_quality_pass=True)]
        self.tracker.update(det, frame_idx=1, is_detection_frame=True)

        # Simulate 6 missing frames (> exit_patience_frames=5)
        for f in range(2, 8):
            tracks = self.tracker.update([], frame_idx=f, is_detection_frame=False)

        # Track should be pruned
        self.assertEqual(len(self.tracker.tracks), 0)


if __name__ == "__main__":
    unittest.main()
