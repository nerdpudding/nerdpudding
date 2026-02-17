"""Standalone test: frame capture + sliding window.

Usage:
    cd video_chat
    python -m scripts.test_capture                          # webcam (device 0)
    python -m scripts.test_capture --source test_files/videos/test.mp4  # video file
    python -m scripts.test_capture --source 2               # specific device ID

Captures frames for a few seconds, then reports what's in the sliding window.
"""

import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test frame capture and sliding window")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: device ID (int) or file path (str). Default: 0 (webcam)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="How long to capture in seconds. Default: video duration for files, 5s for live sources",
    )
    args = parser.parse_args()

    # Parse source: try int first (device ID), fall back to string (file path)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    import cv2

    from app.sliding_window import SlidingWindow
    from app.frame_capture import FrameCapture

    # For video files, auto-detect duration if not specified
    duration = args.duration
    if isinstance(source, str) and duration is None:
        probe = cv2.VideoCapture(source)
        fps = probe.get(cv2.CAP_PROP_FPS)
        total = probe.get(cv2.CAP_PROP_FRAME_COUNT)
        probe.release()
        if fps > 0 and total > 0:
            duration = total / fps
            logger.info(f"Auto-detected video duration: {duration:.1f}s")
    if duration is None:
        duration = 5.0

    window = SlidingWindow()
    capture = FrameCapture(on_frame=window.push)

    logger.info(f"Starting capture from: {source}")
    capture.start(source)

    logger.info(f"Capturing for {duration:.1f}s...")
    time.sleep(duration)

    capture.stop()

    frames = window.get_frames()
    logger.info(f"Frames in window: {window.count}")
    if frames:
        import os

        out_dir = "test_files/videos/capture_test"
        os.makedirs(out_dir, exist_ok=True)

        for i, frame in enumerate(frames):
            path = os.path.join(out_dir, f"frame_{i:03d}.jpg")
            frame.save(path)

        logger.info(
            f"Saved {len(frames)} frames to {out_dir}/ "
            f"(size: {frames[0].size})"
        )
    else:
        logger.warning("No frames captured!")


if __name__ == "__main__":
    main()
