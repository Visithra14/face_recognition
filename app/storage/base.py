from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class FaceRecord:
    face_id: str
    embedding: Optional[np.ndarray]
    first_seen: str
    last_seen: str
    visit_count: int = 1
    first_crop_path: Optional[str] = None

    @property
    def total_visits(self) -> int:
        return self.visit_count


@dataclass
class EventRecord:
    face_id: str
    event_type: str  # "entry" or "exit"
    timestamp: str
    image_path: str
    track_id: int
    event_id: Optional[int] = None

    @property
    def crop_path(self) -> str:
        return self.image_path

    @property
    def frame_idx(self) -> int:
        return self.track_id


class DatabaseRepository(ABC):
    """Abstract interface defining all data persistence contracts for the system."""

    @abstractmethod
    def init_db(self) -> None:
        """Initializes database schema, indexes, and performance PRAGMAs."""
        pass

    @abstractmethod
    def register_face(
        self,
        face_id: str,
        embedding: np.ndarray,
        timestamp: str,
        crop_path: Optional[str] = None
    ) -> bool:
        """Registers a novel facial identity into the persistent store."""
        pass

    @abstractmethod
    def update_face_last_seen(self, face_id: str, timestamp: str) -> None:
        """Updates last-seen timestamp and visit count upon re-identification."""
        pass

    @abstractmethod
    def increment_visit_count(self, face_id: str) -> None:
        """Increments visit counter when an existing identity enters on a subsequent visit."""
        pass

    @abstractmethod
    def log_event(
        self,
        face_id: str,
        event_type: str,
        timestamp: str,
        image_path: str,
        track_id: int
    ) -> None:
        """Persists an ENTRY or EXIT event record."""
        pass

    @abstractmethod
    def get_all_faces(self) -> List[FaceRecord]:
        """Loads all registered identities and deserialized embeddings for gallery hydration."""
        pass

    @abstractmethod
    def get_unique_visitor_count(self) -> int:
        """Returns the total count of unique visitors (distinct face_ids in faces table)."""
        pass

    @abstractmethod
    def get_recent_events(self, limit: int = 50) -> List[EventRecord]:
        """Retrieves most recent lifecycle events for UI / audit display."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Gracefully closes database connection and checkpoints pending WAL pages."""
        pass
