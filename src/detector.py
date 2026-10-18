import os
import urllib.request
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class FaceDetector:
    """
    Real-time face detector utilizing YOLOv8 Face model.
    Falls back gracefully to OpenCV YuNet face detector if YOLO weights are unavailable.
    """
    def __init__(
        self,
        model_name: str = "yolov8n-face.pt",
        confidence_threshold: float = 0.50,
        device: str = "cuda:0",
        min_face_size: Tuple[int, int] = (40, 40)
    ):
        self.model_name = model_name
        self.conf_threshold = confidence_threshold
        self.device = device
        self.min_face_size = min_face_size
        self.model = None
        self.use_yolo = False

        self._init_detector()

    def _init_detector(self) -> None:
        """Initializes the YOLOv8 face model, downloading pre-trained weights if needed."""
        try:
            from ultralytics import YOLO
            import torch

            # Determine device availability
            actual_device = self.device
            if "cuda" in self.device and not torch.cuda.is_available():
                actual_device = "cpu"

            weights_path = os.path.join("models", self.model_name)
            os.makedirs("models", exist_ok=True)

            # Auto-download lightweight yolov8n-face weights if not present locally
            if not os.path.exists(weights_path):
                print(f"[Detector] Downloading YOLOv8 face model to {weights_path}...")
                url = "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"
                try:
                    urllib.request.urlretrieve(url, weights_path)
                    print("[Detector] Download complete.")
                except Exception as dl_err:
                    print(f"[Detector] Could not download yolov8n-face ({dl_err}), using default yolov8n.pt...")
                    weights_path = "yolov8n.pt"

            self.model = YOLO(weights_path)
            self.model.to(actual_device)
            self.use_yolo = True
            print(f"[Detector] YOLOv8 Face detector initialized on {actual_device}.")
        except Exception as e:
            print(f"[Detector] YOLOv8 initialization notice: {e}. Utilizing OpenCV native face detector.")
            self.use_yolo = False

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in a frame.
        Returns list of dicts: [{'bbox': (x1, y1, x2, y2), 'confidence': float, 'crop': np.ndarray}]
        """
        h, w = frame.shape[:2]
        detections = []

        if self.use_yolo and self.model is not None:
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device if "cuda" in self.device else "cpu"
            )
            for res in results:
                boxes = res.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])

                    # Clamp to frame dimensions
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))

                    bw = x2 - x1
                    bh = y2 - y1
                    if bw >= self.min_face_size[0] and bh >= self.min_face_size[1]:
                        crop = frame[y1:y2, x1:x2].copy()
                        detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "confidence": conf,
                            "crop": crop
                        })
        else:
            # Fallback using OpenCV Haar Cascade or basic detector if YOLO not loaded yet
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=self.min_face_size)
            for (x, y, bw, bh) in faces:
                x1, y1, x2, y2 = x, y, x + bw, y + bh
                crop = frame[y1:y2, x1:x2].copy()
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": 0.85,
                    "crop": crop
                })

        return detections
