import threading
from typing import Tuple, Optional, Dict, List
import numpy as np

from app.storage.base import DatabaseRepository
from app.config import RecognitionConfig
from app.logger import get_logger


class IdentityManager:
    """
    Biometric Gallery & Auto-Registration Engine.
    Maintains an in-memory matrix of L2-normalized 512-d embeddings for sub-millisecond
    vectorized cosine similarity matching. Automatically registers novel faces and hydrates
    from the persistent database repository on startup.
    """
    def __init__(self, repo: DatabaseRepository, config: RecognitionConfig):
        self.repo = repo
        self.config = config
        self.logger = get_logger("IdentityManager")

        self._lock = threading.Lock()
        self.gallery: Dict[str, np.ndarray] = {}  # {face_id: 512-d embedding}
        self.id_list: List[str] = []
        self.matrix_cache: Optional[np.ndarray] = None  # (M, 512)

        self._hydrate_from_db()

    def _hydrate_from_db(self) -> None:
        """Loads all existing registered identities from database into memory."""
        with self._lock:
            existing_faces = self.repo.get_all_faces()
            for rec in existing_faces:
                if rec.embedding is not None and rec.embedding.shape == (512,):
                    norm = np.linalg.norm(rec.embedding)
                    self.gallery[rec.face_id] = rec.embedding / (norm + 1e-6) if norm > 1e-6 else rec.embedding

            self._rebuild_matrix_cache()
            self.logger.info(f"Hydrated {len(self.gallery)} existing visitor identities from database.")

    def _rebuild_matrix_cache(self) -> None:
        """Rebuilds the contiguous (M, 512) matrix cache for fast NumPy BLAS dot-product multiplication."""
        self.id_list = list(self.gallery.keys())
        if self.id_list:
            self.matrix_cache = np.vstack([self.gallery[fid] for fid in self.id_list]).astype(np.float32)
        else:
            self.matrix_cache = None

    def _generate_next_id(self) -> str:
        """Generates the next sequential unique identifier (e.g. VISITOR_001)."""
        existing_count = len(self.gallery)
        candidate_num = existing_count + 1
        candidate_id = f"VISITOR_{candidate_num:03d}"

        # Guarantee zero collision
        while candidate_id in self.gallery:
            candidate_num += 1
            candidate_id = f"VISITOR_{candidate_num:03d}"
        return candidate_id

    def match_or_register(
        self,
        embedding: np.ndarray,
        timestamp: str,
        crop_path: Optional[str] = None
    ) -> Tuple[str, bool, float]:
        """
        Matches a 512-d candidate embedding against the registered gallery using cosine similarity.
        Returns: (face_id: str, is_new: bool, max_similarity: float)
        """
        if embedding is None:
            # Fallback if crop could not produce embedding
            new_id = self._generate_next_id()
            return new_id, True, 0.0

        # L2-normalize query embedding
        norm = np.linalg.norm(embedding)
        norm_query = (embedding / (norm + 1e-6)).astype(np.float32)
        max_sim = 0.0

        with self._lock:
            if self.matrix_cache is not None and len(self.id_list) > 0:
                # Vectorized cosine similarity: (M, 512) dot (512,) -> (M,)
                sims = np.dot(self.matrix_cache, norm_query)
                best_idx = int(np.argmax(sims))
                max_sim = float(sims[best_idx])

                if max_sim >= self.config.similarity_threshold:
                    matched_id = self.id_list[best_idx]
                    self.repo.update_face_last_seen(matched_id, timestamp)
                    self.logger.info(f"[RE-ID] Recognized {matched_id} (Cosine Similarity: {max_sim:.3f})")
                    return matched_id, False, max_sim

            # No match found above threshold -> Auto-register novel visitor
            new_id = self._generate_next_id()
            self.gallery[new_id] = norm_query
            self._rebuild_matrix_cache()

            # Persist to database
            self.repo.register_face(new_id, norm_query, timestamp, crop_path)
            self.logger.info(f"[AUTO-REGISTER] Registered new identity: {new_id} (Best Sim: {max_sim:.3f})")
            return new_id, True, max_sim

    def get_unique_visitor_count(self) -> int:
        """Returns total distinct registered visitor count."""
        with self._lock:
            return len(self.gallery)
