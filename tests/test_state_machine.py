import os
import shutil
import unittest
import numpy as np

from app.config import StorageConfig, TrackingConfig
from app.storage.sqlite_repo import SQLiteRepository
from app.tracker import KalmanTracklet
from app.events import EventManager, VisitorState


class TestStateMachine(unittest.TestCase):
    def setUp(self):
        self.test_dir = "logs/test_phase6"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_path = os.path.join(self.test_dir, "test_events.db")
        self.repo = SQLiteRepository(self.db_path)
        self.storage_cfg = StorageConfig(
            db_path=self.db_path,
            log_dir=self.test_dir
        )
        self.tracking_cfg = TrackingConfig(min_confirm_frames=3, exit_patience_frames=3)
        self.event_mgr = EventManager(self.repo, self.storage_cfg, self.tracking_cfg)

    def tearDown(self):
        self.repo.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_atomic_image_save(self):
        dummy_crop = np.zeros((60, 60, 3), dtype=np.uint8)
        dummy_crop[10:50, 10:50] = (255, 200, 100)

        path = self.event_mgr._atomic_save_crop(dummy_crop, "VISITOR_001", "entry", "2026-10-18T16:00:00")
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith(".jpg"))

        # Verify no .tmp files were left behind
        dirname = os.path.dirname(path)
        tmp_files = [f for f in os.listdir(dirname) if f.startswith(".tmp_")]
        self.assertEqual(len(tmp_files), 0)

    def test_strict_one_entry_and_one_exit_lifecycle(self):
        dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)

        # 1. Register face in DB so foreign key passes
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        self.repo.register_face("VISITOR_001", emb, "2026-10-18T16:00:00")

        # 2. Simulate Track Appearance across 4 frames
        track = KalmanTracklet(bbox=(10, 10, 50, 50), confidence=0.95)
        track.face_id = "VISITOR_001"
        track.best_crop = dummy_crop

        # Frame 1: 1 hit (< min_confirm_frames=3) -> No event
        track.hits = 1
        self.event_mgr.process_tracks([track], frame_idx=1, timestamp_str="2026-10-18T16:00:01")
        self.assertEqual(len(self.repo.get_recent_events()), 0)

        # Frame 2: 2 hits -> No event
        track.hits = 2
        self.event_mgr.process_tracks([track], frame_idx=2, timestamp_str="2026-10-18T16:00:02")
        self.assertEqual(len(self.repo.get_recent_events()), 0)

        # Frame 3: 3 hits (>= min_confirm_frames=3) -> ENTRY EVENT FIRED
        track.hits = 3
        self.event_mgr.process_tracks([track], frame_idx=3, timestamp_str="2026-10-18T16:00:03")
        events = self.repo.get_recent_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "entry")
        self.assertEqual(events[0].face_id, "VISITOR_001")

        # Frame 4: 4 hits (already INSIDE) -> Zero duplicate entry event!
        track.hits = 4
        self.event_mgr.process_tracks([track], frame_idx=4, timestamp_str="2026-10-18T16:00:04")
        self.assertEqual(len(self.repo.get_recent_events()), 1)

        # Frame 5: Track drops out of active_tracks -> EXIT EVENT FIRED
        self.event_mgr.process_tracks([], frame_idx=5, timestamp_str="2026-10-18T16:00:05")
        events_after_exit = self.repo.get_recent_events()
        self.assertEqual(len(events_after_exit), 2)
        self.assertEqual(events_after_exit[0].event_type, "exit")
        self.assertEqual(events_after_exit[1].event_type, "entry")

    def test_shutdown_flush_active_tracks(self):
        dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        emb = np.random.randn(512).astype(np.float32)
        self.repo.register_face("VISITOR_002", emb, "2026-10-18T16:00:00")

        track = KalmanTracklet(bbox=(20, 20, 60, 60), confidence=0.9)
        track.face_id = "VISITOR_002"
        track.hits = 5
        track.best_crop = dummy_crop

        # Bring track inside
        self.event_mgr.process_tracks([track], frame_idx=10, timestamp_str="2026-10-18T16:00:10")
        self.assertEqual(len(self.repo.get_recent_events()), 1)

        # Trigger flush on stream end
        self.event_mgr.flush_active_tracks(final_frame_idx=15, timestamp_str="2026-10-18T16:00:15")
        events = self.repo.get_recent_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "exit")
        self.assertEqual(len(self.event_mgr.active_sessions), 0)


if __name__ == "__main__":
    unittest.main()
