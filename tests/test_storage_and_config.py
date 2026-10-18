import os
import shutil
import unittest
import numpy as np

from app.config import load_config, AppConfig
from app.logger import setup_logger
from app.storage.sqlite_repo import SQLiteRepository


class TestStorageAndConfig(unittest.TestCase):
    def setUp(self):
        self.test_dir = "logs/test_phase2"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_path = os.path.join(self.test_dir, "test_faces.db")
        self.log_path = os.path.join(self.test_dir, "test_events.log")
        self.repo = SQLiteRepository(self.db_path)
        self.logger = setup_logger("TestLogger", log_file=self.log_path)

    def tearDown(self):
        self.repo.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_config_loader(self):
        cfg = load_config("config.json")
        self.assertIsInstance(cfg, AppConfig)
        self.assertIn(cfg.source.type, ["file", "rtsp"])
        self.assertGreater(cfg.detection_skip_frames, -1)
        self.assertGreater(cfg.quality.min_face_size, 0)
        self.assertIn(cfg.resolve_device(), ["cuda:0", "cpu"])

    def test_sqlite_wal_and_schema(self):
        # Register a face with 512-d embedding
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        registered = self.repo.register_face("VISITOR_001", emb, "2026-10-18T15:00:00", "logs/entries/crop1.jpg")
        self.assertTrue(registered)
        self.assertEqual(self.repo.get_unique_visitor_count(), 1)

        # Re-registering duplicate should fail gracefully
        self.assertFalse(self.repo.register_face("VISITOR_001", emb, "2026-10-18T15:00:00"))
        self.assertEqual(self.repo.get_unique_visitor_count(), 1)

        # Log entry and exit events
        self.repo.log_event("VISITOR_001", "entry", "2026-10-18T15:00:01", "logs/entries/crop1.jpg", 1)
        self.repo.log_event("VISITOR_001", "exit", "2026-10-18T15:00:10", "logs/exits/crop1.jpg", 1)

        events = self.repo.get_recent_events(limit=10)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "exit")
        self.assertEqual(events[1].event_type, "entry")

        # Hydrate faces and check embedding deserialization
        faces = self.repo.get_all_faces()
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].face_id, "VISITOR_001")
        self.assertEqual(faces[0].embedding.shape, (512,))
        self.assertTrue(np.allclose(faces[0].embedding, emb, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
