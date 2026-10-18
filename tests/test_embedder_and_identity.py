import os
import shutil
import unittest
import numpy as np

from app.config import RecognitionConfig
from app.embedder import FaceEmbedder
from app.storage.sqlite_repo import SQLiteRepository
from app.identity import IdentityManager


class TestEmbedderAndIdentity(unittest.TestCase):
    def setUp(self):
        self.test_dir = "logs/test_phase4"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_path = os.path.join(self.test_dir, "test_identities.db")
        self.repo = SQLiteRepository(self.db_path)
        self.rec_cfg = RecognitionConfig(model_name="buffalo_sc", similarity_threshold=0.50)
        self.embedder = FaceEmbedder(self.rec_cfg, device="auto")
        self.identity_mgr = IdentityManager(self.repo, self.rec_cfg)

    def tearDown(self):
        self.repo.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_embedder_output(self):
        # Synthetic face crop (112, 112, 3)
        dummy_crop = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        emb = self.embedder.embed_crop(dummy_crop)

        self.assertIsNotNone(emb)
        self.assertEqual(emb.shape, (512,))
        self.assertEqual(emb.dtype, np.float32)
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_auto_registration_and_reidentification(self):
        # Vector A: Identity 1
        vec_a = np.random.randn(512).astype(np.float32)
        vec_a /= np.linalg.norm(vec_a)

        # 1. First sight -> Auto-register VISITOR_001
        fid1, is_new1, sim1 = self.identity_mgr.match_or_register(vec_a, "2026-10-18T15:00:00")
        self.assertEqual(fid1, "VISITOR_001")
        self.assertTrue(is_new1)
        self.assertEqual(self.identity_mgr.get_unique_visitor_count(), 1)

        # 2. Re-sight identical vector -> Re-identified as VISITOR_001, is_new=False
        fid2, is_new2, sim2 = self.identity_mgr.match_or_register(vec_a, "2026-10-18T15:00:05")
        self.assertEqual(fid2, "VISITOR_001")
        self.assertFalse(is_new2)
        self.assertAlmostEqual(sim2, 1.0, places=4)
        self.assertEqual(self.identity_mgr.get_unique_visitor_count(), 1)  # Zero double count!

        # 3. Vector B: Distinct orthogonal identity
        vec_b = -vec_a  # Opposite on hypersphere (similarity = -1.0)
        fid3, is_new3, sim3 = self.identity_mgr.match_or_register(vec_b, "2026-10-18T15:00:10")
        self.assertEqual(fid3, "VISITOR_002")
        self.assertTrue(is_new3)
        self.assertEqual(self.identity_mgr.get_unique_visitor_count(), 2)

    def test_database_hydration_on_restart(self):
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)

        self.identity_mgr.match_or_register(vec, "2026-10-18T15:00:00")
        self.assertEqual(self.identity_mgr.get_unique_visitor_count(), 1)

        # Re-instantiate IdentityManager using the same repository
        new_mgr = IdentityManager(self.repo, self.rec_cfg)
        self.assertEqual(new_mgr.get_unique_visitor_count(), 1)
        self.assertIn("VISITOR_001", new_mgr.gallery)


if __name__ == "__main__":
    unittest.main()
