import os
import sys
import time
import signal
from datetime import datetime
from typing import Optional, List
import numpy as np
import cv2

from app.config import AppConfig
from app.storage.sqlite_repo import SQLiteRepository
from app.video_source import VideoSource
from app.detector import YOLOFaceDetector
from app.embedder import FaceEmbedder
from app.identity import IdentityManager
from app.tracker import ByteFaceTracker, KalmanTracklet
from app.events import EventManager
from app.logger import get_logger


class FaceTrackerPipeline:
    """
    Master Computer Vision Pipeline Orchestrator.
    Connects ingestion, detection, tracking, biometric recognition, state machine,
    and output rendering into an error-resilient real-time application.
    """
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger("Pipeline")
        self.is_interrupted = False

        # Initialize Storage & Repositories
        self.repo = SQLiteRepository(self.config.storage.db_path)

        # Initialize Core Modules
        resolved_dev = self.config.resolve_device()
        self.logger.info(f"Target Compute Device resolved to: {resolved_dev}")

        self.source = VideoSource(self.config.source)
        self.detector = YOLOFaceDetector(self.config.detector, self.config.quality, device=resolved_dev)
        self.embedder = FaceEmbedder(self.config.recognition, device=resolved_dev)
        self.identity_mgr = IdentityManager(self.repo, self.config.recognition)
        self.tracker = ByteFaceTracker(
            self.config.tracking,
            reverify_every_n_frames=self.config.recognition.reverify_every_n_frames
        )
        self.event_mgr = EventManager(self.repo, self.config.storage, self.config.tracking)

        # Video Writer for Output Demonstration
        self.video_writer: Optional[cv2.VideoWriter] = None
        if self.config.display.save_output_video:
            out_path = os.path.join(self.config.storage.log_dir, "output_annotated.mp4")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                out_path, fourcc, self.source.fps or 25.0, (self.source.width, self.source.height)
            )
            self.logger.info(f"Annotated video output will be saved to: {out_path}")

        # Register Signal Handlers for Graceful Shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Intercepts termination signals to guarantee audit integrity."""
        self.logger.info(f"Termination signal ({sig}) received. Initiating graceful shutdown...")
        self.is_interrupted = True

    def _render_hud_overlay(
        self,
        frame: np.ndarray,
        tracks: List[KalmanTracklet],
        current_fps: float,
        frame_idx: int
    ) -> np.ndarray:
        """Renders bounding boxes, identity tags, and status bar directly onto the frame."""
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Top HUD Banner (Semi-transparent dark glass)
        banner_h = 55
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 18, 28), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.line(canvas, (0, banner_h), (w, banner_h), (56, 189, 248), 2)

        # 2. Text Indicators
        font = cv2.FONT_HERSHEY_SIMPLEX
        is_rtsp = self.source.is_rtsp
        mode_label = "[LIVE RTSP STREAM]" if is_rtsp else "[VIDEO FILE]"
        cv2.putText(canvas, mode_label, (15, 35), font, 0.65, (52, 211, 153), 2, cv2.LINE_AA)

        unique_cnt = self.identity_mgr.get_unique_visitor_count()
        vis_text = f"UNIQUE VISITORS: {unique_cnt}"
        cv2.putText(canvas, vis_text, (250, 35), font, 0.75, (56, 189, 248), 2, cv2.LINE_AA)

        # Active In-Frame count
        in_frame_cnt = len([t for t in tracks if t.time_since_update == 0])
        cv2.putText(canvas, f"In-Frame: {in_frame_cnt}", (540, 35), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # FPS and Frame Skip
        fps_info = f"FPS: {current_fps:.1f} (Skip: {self.config.detection_skip_frames})"
        cv2.putText(canvas, fps_info, (max(700, w - 260), 35), font, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

        # 3. Draw Bounding Boxes and Identity Badges
        for t in tracks:
            if t.time_since_update != 0:
                continue  # Only draw confirmed active detections on current frame

            x1, y1, x2, y2 = t.bbox
            is_identified = t.face_id is not None
            box_color = (56, 189, 248) if is_identified else (148, 163, 184)  # Cyan if identified, gray if tentative

            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

            label = f"{t.face_id or 'TENTATIVE'} [#{t.track_id}]"
            if t.match_similarity > 0:
                label += f" ({t.match_similarity:.2f})"

            (lw, lh), _ = cv2.getTextSize(label, font, 0.50, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - lh - 8)), (x1 + lw + 6, y1), box_color, -1)
            cv2.putText(canvas, label, (x1 + 3, y1 - 4), font, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

        return canvas

    def run(self) -> int:
        """
        Executes the main pipeline loop.
        Returns final unique visitor count upon completion.
        """
        self.logger.info("=" * 65)
        self.logger.info("PIPELINE START: Intelligent Face Tracker & Unique Visitor Counter")
        self.logger.info(f"Target: {self.config.source.get_effective_source()}")
        self.logger.info(f"Frame Skip Interval: {self.config.detection_skip_frames} frames")
        self.logger.info("=" * 65)

        skip_interval = self.config.detection_skip_frames
        fps_history: List[float] = []
        start_time = time.time()
        frame_idx = 0

        try:
            for frame, idx in self.source.stream_frames():
                if self.is_interrupted:
                    break

                t0 = time.time()
                frame_idx = idx
                now_str = datetime.now().isoformat()
                is_detect_frame = (frame_idx % (skip_interval + 1) == 0) or (frame_idx == 1)

                try:
                    # 1. Detection Phase
                    if is_detect_frame:
                        detections = self.detector.detect(frame)
                        tracks = self.tracker.update(detections, frame_idx, is_detection_frame=True)
                    else:
                        tracks = self.tracker.update([], frame_idx, is_detection_frame=False)

                    # 2. Recognition & Identity Binding Phase
                    needing_embed = self.tracker.get_tracks_needing_embedding(frame_idx)
                    for t in needing_embed:
                        if t.best_crop is not None:
                            emb = self.embedder.embed_crop(t.best_crop)
                            if emb is not None:
                                face_id, is_new, sim = self.identity_mgr.match_or_register(emb, now_str)
                                self.tracker.bind_identity(t.track_id, face_id, sim, frame_idx)

                    # 3. State Machine & Lifecycle Auditing Phase
                    self.event_mgr.process_tracks(tracks, frame_idx, now_str)

                except Exception as frame_err:
                    self.logger.error(f"Error processing frame #{frame_idx}: {frame_err}", exc_info=True)
                    continue

                # 4. Metric & HUD Rendering
                dt = time.time() - t0
                instant_fps = 1.0 / dt if dt > 0 else 30.0
                fps_history.append(instant_fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)

                annotated_frame = self._render_hud_overlay(frame, tracks, avg_fps, frame_idx)

                # 5. Output Handling
                if self.video_writer is not None:
                    self.video_writer.write(annotated_frame)

                if self.config.display.show_overlay:
                    cv2.imshow(self.config.display.window_name, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        self.logger.info("User requested stop via keyboard.")
                        break

        except Exception as pipe_err:
            self.logger.error(f"Fatal pipeline error: {pipe_err}", exc_info=True)
        finally:
            self._shutdown(frame_idx, start_time)

        return self.identity_mgr.get_unique_visitor_count()

    def _shutdown(self, final_frame_idx: int, start_time: float) -> None:
        """Executes graceful shutdown: flushes active tracks, syncs WAL, and closes sinks."""
        self.logger.info("Shutting down pipeline and finalizing active visitor sessions...")
        now_str = datetime.now().isoformat()

        try:
            self.event_mgr.flush_active_tracks(final_frame_idx, now_str)
        except Exception as e:
            self.logger.error(f"Error during exit flush: {e}")

        try:
            self.source.release()
        except Exception:
            pass

        if self.video_writer is not None:
            try:
                self.video_writer.release()
                self.logger.info("Annotated demonstration video finalized.")
            except Exception:
                pass

        if self.config.display.show_overlay:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        total_elapsed = time.time() - start_time
        final_count = self.identity_mgr.get_unique_visitor_count()
        self.repo.close()

        self.logger.info("=" * 65)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info(f"Total Frames Processed  : {final_frame_idx}")
        self.logger.info(f"Total Processing Time   : {total_elapsed:.2f}s (Avg FPS: {final_frame_idx / max(1e-3, total_elapsed):.1f})")
        self.logger.info(f"TOTAL UNIQUE VISITORS   : {final_count}")
        self.logger.info(f"Database Path           : {os.path.abspath(self.config.storage.db_path)}")
        self.logger.info(f"Audit Log File          : {os.path.abspath(self.config.storage.events_log_path)}")
        self.logger.info("=" * 65)
