from dataclasses import dataclass, field
import json
import os
from typing import Optional


@dataclass
class SourceConfig:
    type: str = "file"  # "file" or "rtsp"
    path: str = "data/video_sample1.mp4"
    rtsp_url: str = ""

    def get_effective_source(self) -> str:
        """Returns the active stream target depending on source type."""
        if self.type.lower() == "rtsp" and self.rtsp_url:
            return self.rtsp_url
        return self.path


@dataclass
class DetectorConfig:
    model_path: str = "models/yolov8n-face.pt"
    conf_threshold: float = 0.50
    input_size: int = 640


@dataclass
class RecognitionConfig:
    model_name: str = "buffalo_sc"
    similarity_threshold: float = 0.45
    reverify_every_n_frames: int = 30


@dataclass
class TrackingConfig:
    algorithm: str = "bytetrack"
    min_confirm_frames: int = 5
    exit_patience_frames: int = 30


@dataclass
class QualityConfig:
    min_face_size: int = 40
    blur_threshold: float = 50.0  # Laplacian variance threshold


@dataclass
class StorageConfig:
    db_path: str = "data/faces.db"
    log_dir: str = "logs"

    @property
    def events_log_path(self) -> str:
        return os.path.join(self.log_dir, "events.log")

    @property
    def entries_dir(self) -> str:
        return os.path.join(self.log_dir, "entries")

    @property
    def exits_dir(self) -> str:
        return os.path.join(self.log_dir, "exits")


@dataclass
class DisplayConfig:
    show_overlay: bool = True
    save_output_video: bool = True
    window_name: str = "Intelligent Face Tracker & Unique Visitor Counter"


@dataclass
class AppConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    detection_skip_frames: int = 3
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    device: str = "auto"
    display: DisplayConfig = field(default_factory=DisplayConfig)

    def resolve_device(self) -> str:
        """Resolves 'auto' to 'cuda:0' if an NVIDIA GPU is available, else 'cpu'."""
        if self.device.lower() == "auto":
            try:
                import torch
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device


def load_config(config_path: str = "config.json") -> AppConfig:
    """
    Loads and validates config.json into strongly-typed AppConfig dataclass.
    Falls back to safe defaults if fields are missing.
    """
    if not os.path.exists(config_path):
        print(f"[Config] Warning: {config_path} not found. Creating default configuration.")
        cfg = AppConfig()
        save_config(cfg, config_path)
        return cfg

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Parse nested structures safely
    source_cfg = SourceConfig(**raw.get("source", {}))
    detector_cfg = DetectorConfig(**raw.get("detector", {}))
    recog_cfg = RecognitionConfig(**raw.get("recognition", {}))
    track_cfg = TrackingConfig(**raw.get("tracking", {}))
    quality_cfg = QualityConfig(**raw.get("quality", {}))
    storage_cfg = StorageConfig(**raw.get("storage", {}))
    display_cfg = DisplayConfig(**raw.get("display", {}))

    return AppConfig(
        source=source_cfg,
        detection_skip_frames=int(raw.get("detection_skip_frames", 3)),
        detector=detector_cfg,
        recognition=recog_cfg,
        tracking=track_cfg,
        quality=quality_cfg,
        storage=storage_cfg,
        device=str(raw.get("device", "auto")),
        display=display_cfg
    )


def save_config(config: AppConfig, config_path: str = "config.json") -> None:
    """Serializes AppConfig instance to disk as formatted JSON."""
    data = {
        "source": {
            "type": config.source.type,
            "path": config.source.path,
            "rtsp_url": config.source.rtsp_url
        },
        "detection_skip_frames": config.detection_skip_frames,
        "detector": {
            "model_path": config.detector.model_path,
            "conf_threshold": config.detector.conf_threshold,
            "input_size": config.detector.input_size
        },
        "recognition": {
            "model_name": config.recognition.model_name,
            "similarity_threshold": config.recognition.similarity_threshold,
            "reverify_every_n_frames": config.recognition.reverify_every_n_frames
        },
        "tracking": {
            "algorithm": config.tracking.algorithm,
            "min_confirm_frames": config.tracking.min_confirm_frames,
            "exit_patience_frames": config.tracking.exit_patience_frames
        },
        "quality": {
            "min_face_size": config.quality.min_face_size,
            "blur_threshold": config.quality.blur_threshold
        },
        "storage": {
            "db_path": config.storage.db_path,
            "log_dir": config.storage.log_dir
        },
        "device": config.device,
        "display": {
            "show_overlay": config.display.show_overlay,
            "save_output_video": config.display.save_output_video
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
