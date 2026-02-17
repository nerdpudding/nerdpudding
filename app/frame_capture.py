import io
import logging
import threading
import time
from collections import deque
from typing import Callable, Optional, Tuple

import cv2
from PIL import Image

from app.config import CAPTURE_FPS, FRAME_JPEG_QUALITY

logger = logging.getLogger(__name__)

# Max seconds of JPEG frames to keep for delayed MJPEG display.
# At 30 FPS source, 15 seconds = 450 entries, ~30-45 MB of JPEG data.
_DISPLAY_BUFFER_SECONDS = 15


class FrameCapture:
    """Captures frames from a video source in a background thread.

    Two output paths:
    - **Display buffer**: every frame JPEG-encoded at native rate for smooth
      MJPEG streaming to the browser. Stored as (timestamp, jpeg_bytes).
    - **Inference callback** (on_frame → SlidingWindow): only at CAPTURE_FPS
      rate, for the AI model. Lower rate saves image tokens / VRAM.

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
        # Display buffer: (wall-clock timestamp, JPEG bytes)
        self._display_buffer: deque[Tuple[float, bytes]] = deque()
        self._display_lock = threading.Lock()
        self._src_fps: float = 0

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

    @property
    def source_fps(self) -> float:
        return self._src_fps

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

        self._src_fps = self._capture.get(cv2.CAP_PROP_FPS) or 0
        is_file = isinstance(source, str)
        display_fps = self._src_fps if (is_file and self._src_fps > 0) else 30
        maxlen = int(display_fps * _DISPLAY_BUFFER_SECONDS)
        with self._display_lock:
            self._display_buffer = deque(maxlen=maxlen)

        logger.info(
            f"Opened video source: {source} "
            f"(source FPS: {self._src_fps:.1f}, inference FPS: {CAPTURE_FPS}, "
            f"display buffer: {maxlen} frames)"
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

    def get_display_jpeg(self, target_time: Optional[float] = None) -> Optional[bytes]:
        """Get a JPEG frame from the display buffer.

        Args:
            target_time: Wall-clock timestamp to match. Returns the frame
                closest to this time. None returns the latest frame.

        Returns:
            JPEG bytes, or None if buffer is empty.
        """
        with self._display_lock:
            if not self._display_buffer:
                return None
            if target_time is None:
                return self._display_buffer[-1][1]
            # Edge cases
            if target_time <= self._display_buffer[0][0]:
                return self._display_buffer[0][1]
            if target_time >= self._display_buffer[-1][0]:
                return self._display_buffer[-1][1]
            # Linear scan (buffer is bounded, typically < 500 items)
            best_jpeg = self._display_buffer[0][1]
            best_diff = abs(self._display_buffer[0][0] - target_time)
            for ts, jpeg in self._display_buffer:
                diff = abs(ts - target_time)
                if diff < best_diff:
                    best_jpeg = jpeg
                    best_diff = diff
                elif diff > best_diff:
                    break  # timestamps are sorted, won't get better
            return best_jpeg

    def _capture_loop(self) -> None:
        """Background loop: reads every frame at native rate.

        Pushes every frame to the display buffer (for MJPEG).
        Pushes to SlidingWindow (via on_frame) at CAPTURE_FPS rate only.
        """
        is_file = isinstance(self._source, str)
        src_fps = self._src_fps
        # For video files: pace at source FPS. For live: no extra sleep.
        frame_interval = (1.0 / src_fps) if (is_file and src_fps > 0) else 0
        inference_interval = 1.0 / CAPTURE_FPS
        last_inference_push = 0.0

        while self._running:
            t0 = time.monotonic()

            ret, frame = self._capture.read()

            if not ret:
                if is_file:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.debug("Video file looped")
                    continue
                else:
                    logger.warning("Failed to read frame from source")
                    time.sleep(0.1)
                    continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            with self._lock:
                self._latest_frame = pil_image

            # Display buffer: JPEG-encode every frame
            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG", quality=FRAME_JPEG_QUALITY)
            now = time.time()
            with self._display_lock:
                self._display_buffer.append((now, buf.getvalue()))

            # Inference callback: only at CAPTURE_FPS rate
            mono_now = time.monotonic()
            if mono_now - last_inference_push >= inference_interval:
                last_inference_push = mono_now
                if self._on_frame is not None:
                    self._on_frame(pil_image)

            # Pace video file playback to real-time
            if frame_interval > 0:
                elapsed = time.monotonic() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
