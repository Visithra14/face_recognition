import os
import sys
import shutil
import argparse

from app.config import load_config
from app.pipeline import FaceTrackerPipeline


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Intelligent Face Tracker with Auto-Registration and Unique Visitor Counting"
    )
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--source", type=str, default=None, help="Override video input file path or RTSP URL")
    parser.add_argument("--rtsp", type=str, default=None, help="Directly specify live RTSP camera stream URL")
    parser.add_argument("--skip", type=int, default=None, help="Override detection_skip_frames")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without OpenCV GUI window")
    parser.add_argument("--reset", action="store_true", help="Wipe database, logs, and crops before starting a fresh run")
    return parser.parse_args()


def perform_reset(config) -> None:
    """Wipes previous database, logs, and cropped images for a clean benchmark run."""
    db_path = config.storage.db_path
    log_path = config.storage.events_log_path
    entries_dir = config.storage.entries_dir
    exits_dir = config.storage.exits_dir

    for file_path in [db_path, log_path]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[Reset] Warning: could not remove {file_path}: {e}")

    for folder in [entries_dir, exits_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)

    print("[Reset] Cleared existing database, event logs, and cropped image snapshots.")


def main():
    args = parse_arguments()
    config = load_config(args.config)

    # 1. Handle Reset Flag
    if args.reset:
        perform_reset(config)

    # 2. Apply Command-Line Overrides
    if args.rtsp:
        config.source.type = "rtsp"
        config.source.rtsp_url = args.rtsp
    elif args.source:
        if args.source.startswith(("rtsp://", "http://")):
            config.source.type = "rtsp"
            config.source.rtsp_url = args.source
        else:
            config.source.type = "file"
            config.source.path = args.source

    if args.skip is not None:
        config.detection_skip_frames = args.skip

    if args.no_display:
        config.display.show_overlay = False

    # 3. Instantiate and Run Pipeline
    pipeline = FaceTrackerPipeline(config)
    unique_visitors = pipeline.run()

    print(f"\n[Execution Complete] Final Unique Visitor Count: {unique_visitors}")
    sys.exit(0)


if __name__ == "__main__":
    main()
