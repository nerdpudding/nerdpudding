import logging
import threading
import time
from typing import Callable, Optional

import cv2
from PIL import Image

from app.config import CAPTURE_FPS

logger = logging.getLogger(__name__)


class FrameCapture:
    """Captures frames from a video source in a background thread.

    Works with webcam device IDs (int) and video file paths (str).
    Video files loop automatically for testing purposes.
    """

    def __init__(self, on_frame: Optional[Callable[[Image.Image], None]] = None):
        self._on_frame = on_frame
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_frame: Optional[Image.Image] = None
        self._lock = threading.Lock()
        self._source = None

    @property
    def latest_frame(self) -> Optional[Image.Image]:
        with self._lock:
            return self._latest_frame

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def source(self):
        return self._source

    def start(self, source) -> None:
        """Start capturing from a video source.

        Args:
            source: Device ID (int, e.g. 0 for webcam) or file path (str).
        """
        if self._running:
            self.stop()

        self._source = source
        self._capture = cv2.VideoCapture(source)

        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        src_fps = self._capture.get(cv2.CAP_PROP_FPS)
        logger.info(
            f"Opened video source: {source} "
            f"(source FPS: {src_fps:.1f}, capture FPS: {CAPTURE_FPS})"
        )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing and release the video source."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._source = None
        logger.info("Frame capture stopped")

    def _capture_loop(self) -> None:
        """Background loop that captures frames at the configured FPS.

        For video files: skips frames to match real-time playback at CAPTURE_FPS.
        For live sources (webcam): read() always returns the latest frame.
        """
        interval = 1.0 / CAPTURE_FPS
        is_file = isinstance(self._source, str)
        src_fps = self._capture.get(cv2.CAP_PROP_FPS) if is_file else 0
        # How many source frames to skip per capture interval
        skip = max(1, int(src_fps / CAPTURE_FPS)) if is_file and src_fps > 0 else 1

        while self._running:
            t0 = time.monotonic()

            # For video files, skip ahead to simulate real-time playback
            if is_file and skip > 1:
                for _ in range(skip - 1):
                    self._capture.grab()

            ret, frame = self._capture.read()

            if not ret:
                if is_file:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.debug("Video file looped")
                    continue
                else:
                    logger.warning("Failed to read frame from source")
                    time.sleep(interval)
                    continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            with self._lock:
                self._latest_frame = pil_image

            if self._on_frame is not None:
                self._on_frame(pil_image)

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
