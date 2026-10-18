import os
import shutil
import unittest
import numpy as np
from src.database import DatabaseManager, SystemLogger
from src.lifecycle_manager import LifecycleManager
from src.tracker import MultiFaceTracker, compute_iou


class TestFaceTrackerPipeline(unittest.TestCase):
    """Automated unit and integration tests verifying all hackathon requirements."""

    def setUp(self):
        self.test_dir = "logs/test_run"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_path = os.path.join(self.test_dir, "test_visitors.db")
        self.log_path = os.path.join(self.test_dir, "test_events.log")
        self.entries_dir = os.path.join(self.test_dir, "entries")
        self.exits_dir = os.path.join(self.test_dir, "exits")

        self.logger = SystemLogger.setup_logger(self.log_path)
        self.db = DatabaseManager(self.db_path)
        self.lifecycle = LifecycleManager(
            db_manager=self.db,
            logger=self.logger,
            entries_dir=self.entries_dir,
            exits_dir=self.exits_dir,
            confirmation_frames=3,
            max_lost_frames=10,
            similarity_threshold=0.55
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_database_and_event_logging(self):
        """Tests visitor registration, entry and exit event persistence."""
        dummy_embedding = np.random.randn(512).astype(np.float32)
        dummy_embedding /= np.linalg.norm(dummy_embedding)

        # 1. Register visitor
        registered = self.db.register_new_visitor("VISITOR_001", dummy_embedding, "2026-10-18T13:00:00")
        self.assertTrue(registered)
        self.assertEqual(self.db.get_unique_visitor_count(), 1)

        # 2. Prevent duplicate visitor_id insertion
        dup = self.db.register_new_visitor("VISITOR_001", dummy_embedding, "2026-10-18T13:00:00")
        self.assertFalse(dup)
        self.assertEqual(self.db.get_unique_visitor_count(), 1)

        # 3. Log ENTRY event
        self.db.log_event("VISITOR_001", "ENTRY", "2026-10-18T13:00:01", 10, confidence=0.95)
        # 4. Log EXIT event
        self.db.log_event("VISITOR_001", "EXIT", "2026-10-18T13:00:15", 350, confidence=0.95)

        events = self.db.get_recent_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event_type"], "EXIT")
        self.assertEqual(events[1]["event_type"], "ENTRY")

    def test_strict_single_entry_and_exit_lifecycle(self):
        """Verifies that an appearance generates EXACTLY one entry and one exit."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        fixed_emb = np.random.randn(512).astype(np.float32)
        fixed_emb /= np.linalg.norm(fixed_emb)

        # Simulate person appearing across 5 frames
        for f in range(1, 6):
            tracks = [{
                "track_id": 1,
                "bbox": (100, 100, 200, 200),
                "embedding": fixed_emb,
                "crop": dummy_crop
            }]
            self.lifecycle.process_frame_tracks(dummy_frame, f, tracks)

        # At frame 5 (> confirmation_frames=3), ENTRY must be logged exactly once
        events = self.db.get_recent_events()
        entry_events = [e for e in events if e["event_type"] == "ENTRY"]
        self.assertEqual(len(entry_events), 1)
        self.assertEqual(self.db.get_unique_visitor_count(), 1)

        # Now simulate track disappearance for 15 frames (> max_lost_frames=10)
        for f in range(6, 22):
            self.lifecycle.process_frame_tracks(dummy_frame, f, [])

        # EXIT must now be logged exactly once
        events = self.db.get_recent_events()
        exit_events = [e for e in events if e["event_type"] == "EXIT"]
        self.assertEqual(len(exit_events), 1)

    def test_reidentification_does_not_increment_unique_count(self):
        """Verifies that re-identifying a known visitor in later frames does not increment unique count."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        known_emb = np.random.randn(512).astype(np.float32)
        known_emb /= np.linalg.norm(known_emb)

        # Encounter 1
        for f in range(1, 5):
            self.lifecycle.process_frame_tracks(dummy_frame, f, [{
                "track_id": 1,
                "bbox": (50, 50, 150, 150),
                "embedding": known_emb,
                "crop": dummy_crop
            }])
        # Exit Encounter 1
        for f in range(5, 20):
            self.lifecycle.process_frame_tracks(dummy_frame, f, [])

        initial_count = self.db.get_unique_visitor_count()
        self.assertEqual(initial_count, 1)

        # Encounter 2: Same person returns (new track ID 2, identical embedding)
        for f in range(21, 26):
            self.lifecycle.process_frame_tracks(dummy_frame, f, [{
                "track_id": 2,
                "bbox": (80, 80, 180, 180),
                "embedding": known_emb,
                "crop": dummy_crop
            }])

        # Unique visitor count MUST REMAIN 1
        subsequent_count = self.db.get_unique_visitor_count()
        self.assertEqual(subsequent_count, 1)

    def test_iou_calculation(self):
        """Tests bounding box IoU accuracy."""
        boxA = (0, 0, 10, 10)
        boxB = (0, 0, 10, 10)
        self.assertAlmostEqual(compute_iou(boxA, boxB), 1.0)

        boxC = (10, 10, 20, 20)
        self.assertAlmostEqual(compute_iou(boxA, boxC), 0.0)


if __name__ == "__main__":
    unittest.main()
