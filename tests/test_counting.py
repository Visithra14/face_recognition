import os
import shutil
import unittest
import numpy as np

from app.storage.sqlite_repo import SQLiteRepository
from app.count import get_unique_count, print_visitor_summary


class TestCountingModule(unittest.TestCase):
    def setUp(self):
        self.test_dir = "logs/test_phase7"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_path = os.path.join(self.test_dir, "test_count.db")
        self.repo = SQLiteRepository(self.db_path)

    def tearDown(self):
        self.repo.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_get_unique_count_empty(self):
        count = get_unique_count(self.db_path)
        self.assertEqual(count, 0)

    def test_get_unique_count_with_registered_faces(self):
        # Register 3 distinct identities
        for i in range(1, 4):
            emb = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            self.repo.register_face(f"VISITOR_{i:03d}", emb, f"2026-10-18T16:00:0{i}")

        count = get_unique_count(self.db_path)
        self.assertEqual(count, 3)

    def test_print_visitor_summary_runs_without_exception(self):
        emb = np.random.randn(512).astype(np.float32)
        self.repo.register_face("VISITOR_001", emb, "2026-10-18T16:00:00")
        self.repo.log_event("VISITOR_001", "entry", "2026-10-18T16:00:01", "logs/entries/crop1.jpg", 1)

        # Should execute cleanly without error
        try:
            print_visitor_summary(self.db_path)
            success = True
        except Exception:
            success = False

        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
