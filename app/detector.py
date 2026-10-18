import os
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

from app.config import DetectorConfig, QualityConfig
from app.logger import get_logger


@dataclass
class DetectionResult:
    """Represents a validated face detection bounding box and candidate crop."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    crop: np.ndarray
    blur_score: float
    is_quality_pass: bool


class YOLOFaceDetector:
    """
    Dedicated Face Detector using YOLOv8-Face architecture.
    Applies multi-stage quality gating:
    1. Confidence thresholding
    2. Bounding box dimension bounds (min_face_size)
    3. Context margin expansion (12% padding)
    4. Laplacian variance blur filtering
    """
    def __init__(
        self,
        detector_cfg: DetectorConfig,
        quality_cfg: QualityConfig,
        device: str = "auto"
    ):
        self.detector_cfg = detector_cfg
        self.quality_cfg = quality_cfg
        self.device = device
        self.logger = get_logger("FaceDetector")
        self.model = None

        self._init_model()

    def _init_model(self) -> None:
        """Loads YOLOv8-face weights, auto-downloading if necessary."""
        from ultralytics import YOLO

        weights_path = self.detector_cfg.model_path
        os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)

        if not os.path.exists(weights_path):
            self.logger.info(f"Model weights not found at {weights_path}. Downloading YOLOv8-Face checkpoint...")
            url = "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"
            try:
                urllib.request.urlretrieve(url, weights_path)
                self.logger.info("YOLOv8-Face checkpoint downloaded successfully.")
            except Exception as e:
                self.logger.error(f"Failed to download YOLOv8-Face: {e}. Falling back to default yolov8n.pt...")
                weights_path = "yolov8n.pt"

        # Determine target device
        target_dev = self.device
        if target_dev == "auto":
            try:
                import torch
                target_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                target_dev = "cpu"

        self.model = YOLO(weights_path)
        self.model.to(target_dev)
        self.logger.info(f"YOLOv8 Face Detector loaded on {target_dev} (Input Size: {self.detector_cfg.input_size}px, Conf: {self.detector_cfg.conf_threshold}).")

    def _compute_laplacian_blur(self, crop: np.ndarray) -> float:
        """Computes variance of the Laplacian operator as a sharpness/blur metric."""
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _expand_bbox_with_margin(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
        margin_ratio: float = 0.12
    ) -> Tuple[int, int, int, int]:
        """Expands bounding box by margin_ratio to capture complete facial contour."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)

        nx1 = max(0, x1 - mx)
        ny1 = max(0, y1 - my)
        nx2 = min(w, x2 + mx)
        ny2 = min(h, y2 + my)
        return (nx1, ny1, nx2, ny2)

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Runs YOLOv8-Face detection on a single frame and applies quality gates.
        Returns list of DetectionResult objects.
        """
        if self.model is None or frame is None:
            return []

        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf=self.detector_cfg.conf_threshold,
            imgsz=self.detector_cfg.input_size,
            verbose=False,
            device=self.device if self.device != "auto" else None
        )

        detections: List[DetectionResult] = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                raw_box = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, raw_box)

                # Clamp raw detection coordinates
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                bw = x2 - x1
                bh = y2 - y1

                # Quality Gate 1: Minimum face dimension
                if bw < self.quality_cfg.min_face_size or bh < self.quality_cfg.min_face_size:
                    continue

                # Expand with margin for facial alignment & embedding
                ex1, ey1, ex2, ey2 = self._expand_bbox_with_margin((x1, y1, x2, y2), (h, w))
                crop = frame[ey1:ey2, ex1:ex2].copy()

                # Quality Gate 2: Laplacian blur score
                blur_score = self._compute_laplacian_blur(crop)
                is_quality_pass = blur_score >= self.quality_cfg.blur_threshold

                detections.append(DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    crop=crop,
                    blur_score=blur_score,
                    is_quality_pass=is_quality_pass
                ))

        return detections
