import unittest
import numpy as np
import cv2

from app.config import SourceConfig, DetectorConfig, QualityConfig
from app.video_source import VideoSource
from app.detector import YOLOFaceDetector, DetectionResult


class TestVideoAndDetector(unittest.TestCase):
    def setUp(self):
        self.src_cfg = SourceConfig(type="file", path="data/video_sample1.mp4")
        self.det_cfg = DetectorConfig(model_path="models/yolov8n-face.pt", conf_threshold=0.40, input_size=640)
        self.qual_cfg = QualityConfig(min_face_size=30, blur_threshold=30.0)

    def test_video_source_file_reading(self):
        source = VideoSource(self.src_cfg)
        self.assertGreater(source.fps, 0)
        self.assertGreater(source.width, 0)
        self.assertGreater(source.height, 0)

        # Read 3 frames
        for _ in range(3):
            ret, frame, idx = source.read_frame()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)
            self.assertEqual(len(frame.shape), 3)

        source.release()

    def test_detector_inference_and_quality_gate(self):
        source = VideoSource(self.src_cfg)
        detector = YOLOFaceDetector(self.det_cfg, self.qual_cfg, device="auto")

        ret, frame, _ = source.read_frame()
        self.assertTrue(ret)

        detections = detector.detect(frame)
        source.release()

        self.assertIsInstance(detections, list)
        self.assertGreater(len(detections), 0)

        det = detections[0]
        self.assertIsInstance(det, DetectionResult)
        self.assertEqual(len(det.bbox), 4)
        self.assertGreaterEqual(det.confidence, self.det_cfg.conf_threshold)
        self.assertIsNotNone(det.crop)
        self.assertGreater(det.crop.shape[0], 0)
        self.assertGreater(det.crop.shape[1], 0)
        self.assertGreaterEqual(det.blur_score, 0.0)


if __name__ == "__main__":
    unittest.main()
