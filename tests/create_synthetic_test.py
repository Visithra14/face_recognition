import os
import cv2
import numpy as np


def generate_synthetic_video(output_path: str = "data/sample_video.mp4", duration_sec: int = 4, fps: int = 25):
    """
    Generates a synthetic MP4 video with synthetic face-like patterns moving across the frame.
    Enables automatic end-to-end pipeline validation and dry-run verification.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps

    # Person 1: Enters at frame 10, exits at frame 60
    # Person 2: Enters at frame 35, exits at frame 85
    for frame_idx in range(total_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Background gradient
        img[:, :] = (30, 30, 40)

        # Draw Person 1
        if 10 <= frame_idx <= 60:
            progress = (frame_idx - 10) / 50.0
            cx = int(80 + progress * 400)
            cy = int(240 + np.sin(progress * np.pi) * 20)
            # Face oval (Skin tone)
            cv2.ellipse(img, (cx, cy), (45, 60), 0, 0, 360, (180, 210, 240), -1)
            # Eyes
            cv2.circle(img, (cx - 15, cy - 12), 6, (60, 40, 20), -1)
            cv2.circle(img, (cx + 15, cy - 12), 6, (60, 40, 20), -1)
            # Nose
            cv2.line(img, (cx, cy - 5), (cx, cy + 10), (140, 160, 190), 2)
            # Mouth
            cv2.ellipse(img, (cx, cy + 25), (15, 6), 0, 0, 180, (80, 80, 180), 2)

        # Draw Person 2 (different tone and trajectory)
        if 35 <= frame_idx <= 85:
            progress = (frame_idx - 35) / 50.0
            cx = int(500 - progress * 350)
            cy = int(180 + np.cos(progress * np.pi) * 15)
            # Face oval
            cv2.ellipse(img, (cx, cy), (40, 55), 0, 0, 360, (160, 190, 225), -1)
            # Eyes
            cv2.circle(img, (cx - 12, cy - 10), 5, (40, 30, 20), -1)
            cv2.circle(img, (cx + 12, cy - 10), 5, (40, 30, 20), -1)
            # Mouth
            cv2.ellipse(img, (cx, cy + 20), (12, 5), 0, 0, 180, (70, 70, 160), 2)

        out.write(img)

    out.release()
    print(f"[Test Generator] Synthetic test video generated at {output_path} ({total_frames} frames, {fps} fps).")


if __name__ == "__main__":
    generate_synthetic_video()
