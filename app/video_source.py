import cv2
import time
import queue
import threading
from typing import Optional, Tuple, Generator
import numpy as np

from app.config import SourceConfig
from app.logger import get_logger


class VideoSource:
    """
    Unified video ingestion engine supporting local files and live RTSP streams.
    For RTSP: Uses a decoupled background thread with a 1-frame ring buffer to eliminate latency lag
    and implements exponential backoff auto-reconnect logic.
    For File: Reads frames sequentially with end-of-file detection.
    """
    def __init__(self, config: SourceConfig, reconnect_timeout: float = 30.0):
        self.config = config
        self.reconnect_timeout = reconnect_timeout
        self.logger = get_logger("VideoSource")

        self.is_rtsp = config.type.lower() == "rtsp" or config.get_effective_source().startswith(("rtsp://", "http://"))
        self.source_target = config.get_effective_source()

        self.cap: Optional[cv2.VideoCapture] = None
        self._stopped = False
        self._frame_queue = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Metadata
        self.fps: float = 30.0
        self.width: int = 1280
        self.height: int = 720
        self.total_frames: int = -1
        self.current_frame_idx: int = 0

        self._connect()

    def _connect(self) -> bool:
        """Connects or reconnects to the video stream."""
        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass

            self.logger.info(f"Connecting to video source: {self.source_target} (RTSP={self.is_rtsp})...")
            self.cap = cv2.VideoCapture(self.source_target)

            if self.is_rtsp:
                # Minimize internal driver buffer to prevent stale frame accumulation
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                self.logger.warning(f"Could not open video stream: {self.source_target}")
                return False

            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_rtsp else -1

            self.logger.info(f"Stream opened successfully: {self.width}x{self.height} @ {self.fps:.1f} FPS (Total: {self.total_frames})")

            if self.is_rtsp and (self._thread is None or not self._thread.is_alive()):
                self._stopped = False
                self._thread = threading.Thread(target=self._rtsp_grabber_loop, daemon=True, name="RTSPGrabber")
                self._thread.start()

            return True

    def _rtsp_grabber_loop(self) -> None:
        """Background thread continuously polling the RTSP stream to discard stale frames."""
        backoff = 1.0
        while not self._stopped:
            if self.cap is None or not self.cap.isOpened():
                self.logger.warning(f"RTSP stream disconnected. Attempting reconnect in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, self.reconnect_timeout)
                if self._connect():
                    backoff = 1.0
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.warning("RTSP packet dropped or frame empty. Reconnecting...")
                time.sleep(0.5)
                self._connect()
                continue

            # Push latest frame to queue, discarding old frame if unconsumed
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self._frame_queue.put_nowait(frame)
                backoff = 1.0
            except queue.Full:
                pass

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """
        Retrieves the next frame.
        Returns: (success: bool, frame: Optional[np.ndarray], frame_index: int)
        """
        if self._stopped:
            return False, None, self.current_frame_idx

        if self.is_rtsp:
            try:
                frame = self._frame_queue.get(timeout=2.0)
                self.current_frame_idx += 1
                return True, frame, self.current_frame_idx
            except queue.Empty:
                return False, None, self.current_frame_idx
        else:
            if self.cap is None or not self.cap.isOpened():
                return False, None, self.current_frame_idx

            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None, self.current_frame_idx

            self.current_frame_idx += 1
            return True, frame, self.current_frame_idx

    def stream_frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Generator yielding (frame, frame_idx) until stream terminates."""
        while not self._stopped:
            ret, frame, idx = self.read_frame()
            if not ret or frame is None:
                if not self.is_rtsp:
                    break
                time.sleep(0.1)
                continue
            yield frame, idx

    def release(self) -> None:
        """Stops background threads and releases camera capture."""
        self._stopped = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        self.logger.info("VideoSource released cleanly.")
