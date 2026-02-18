import io
import logging
import threading
import time
import urllib.request
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

    Supported sources:
    - Webcam device IDs (int, e.g. 0)
    - Video file paths (str, loop automatically)
    - RTSP URLs (str, via OpenCV/FFmpeg)
    - HTTP MJPEG streams (str, multipart/x-mixed-replace, custom reader)
    - HTTP video URLs (str, via OpenCV/FFmpeg)
    """

    def __init__(self, on_frame: Optional[Callable[[Image.Image], None]] = None):
        self._on_frame = on_frame
        self._capture: Optional[cv2.VideoCapture] = None
        self._http_response = None  # for HTTP MJPEG streams
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_frame: Optional[Image.Image] = None
        self._lock = threading.Lock()
        self._source = None
        self._is_http_mjpeg = False
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

    @staticmethod
    def _is_http_url(source) -> bool:
        return isinstance(source, str) and source.lower().startswith(("http://", "https://"))

    @staticmethod
    def _probe_http_content_type(url: str) -> str:
        """Probe an HTTP URL to determine its Content-Type."""
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.headers.get("Content-Type", "")
        except Exception:
            # HEAD not supported or failed — try a GET and read minimally
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.headers.get("Content-Type", "")
            except Exception:
                return ""

    def start(self, source) -> None:
        """Start capturing from a video source.

        Args:
            source: Device ID (int, e.g. 0 for webcam), file path (str),
                    RTSP URL, HTTP MJPEG stream URL, or HTTP video URL.
        """
        if self._running:
            self.stop()

        self._source = source
        self._is_http_mjpeg = False

        # Detect HTTP MJPEG streams by probing Content-Type
        if self._is_http_url(source):
            content_type = self._probe_http_content_type(source)
            if "multipart/x-mixed-replace" in content_type:
                self._is_http_mjpeg = True
                logger.info(
                    f"Detected HTTP MJPEG stream: {source} "
                    f"(Content-Type: {content_type})"
                )

        if self._is_http_mjpeg:
            self._start_http_mjpeg(source)
        else:
            self._start_opencv(source)

    def _start_opencv(self, source) -> None:
        """Start capture via OpenCV (files, devices, RTSP, HTTP video)."""
        self._capture = cv2.VideoCapture(source)

        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        self._src_fps = self._capture.get(cv2.CAP_PROP_FPS) or 0
        is_file = isinstance(source, str) and not source.lower().startswith("rtsp://")
        display_fps = self._src_fps if (is_file and self._src_fps > 0) else 30
        maxlen = int(display_fps * _DISPLAY_BUFFER_SECONDS)
        with self._display_lock:
            self._display_buffer = deque(maxlen=maxlen)

        logger.info(
            f"Opened video source (OpenCV): {source} "
            f"(source FPS: {self._src_fps:.1f}, inference FPS: {CAPTURE_FPS}, "
            f"display buffer: {maxlen} frames)"
        )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _start_http_mjpeg(self, url: str) -> None:
        """Start capture from an HTTP MJPEG stream."""
        # Assume live stream at ~25 FPS for display buffer sizing
        self._src_fps = 25
        maxlen = int(self._src_fps * _DISPLAY_BUFFER_SECONDS)
        with self._display_lock:
            self._display_buffer = deque(maxlen=maxlen)

        logger.info(
            f"Opened video source (HTTP MJPEG): {url} "
            f"(assumed FPS: {self._src_fps:.1f}, inference FPS: {CAPTURE_FPS}, "
            f"display buffer: {maxlen} frames)"
        )

        self._running = True
        self._thread = threading.Thread(
            target=self._mjpeg_http_loop, args=(url,), daemon=True
        )
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
        if self._http_response is not None:
            try:
                self._http_response.close()
            except Exception:
                pass
            self._http_response = None
        self._source = None
        self._is_http_mjpeg = False
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

    def _process_frame(self, pil_image: Image.Image, last_inference_push: float) -> float:
        """Shared frame processing: display buffer + inference callback.

        Returns the updated last_inference_push timestamp.
        """
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
        inference_interval = 1.0 / CAPTURE_FPS
        if mono_now - last_inference_push >= inference_interval:
            last_inference_push = mono_now
            if self._on_frame is not None:
                self._on_frame(pil_image)

        return last_inference_push

    def _capture_loop(self) -> None:
        """Background loop (OpenCV): reads every frame at native rate."""
        is_file = isinstance(self._source, str) and not self._source.lower().startswith("rtsp://")
        src_fps = self._src_fps
        # For video files: pace at source FPS. For live: no extra sleep.
        frame_interval = (1.0 / src_fps) if (is_file and src_fps > 0) else 0
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
            last_inference_push = self._process_frame(pil_image, last_inference_push)

            # Pace video file playback to real-time
            if frame_interval > 0:
                elapsed = time.monotonic() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _mjpeg_http_loop(self, url: str) -> None:
        """Background loop: reads JPEG frames from an HTTP MJPEG stream.

        Parses multipart/x-mixed-replace boundaries, extracts JPEG data,
        and feeds frames through the same processing pipeline as OpenCV.
        """
        last_inference_push = 0.0

        while self._running:
            try:
                logger.info(f"Connecting to HTTP MJPEG stream: {url}")
                req = urllib.request.Request(url)
                self._http_response = urllib.request.urlopen(req, timeout=10)

                content_type = self._http_response.headers.get("Content-Type", "")
                # Extract boundary from Content-Type header
                boundary = b""
                for part in content_type.split(";"):
                    part = part.strip()
                    if part.lower().startswith("boundary="):
                        boundary = ("--" + part.split("=", 1)[1].strip()).encode()
                        break

                if not boundary:
                    logger.error(f"No boundary found in Content-Type: {content_type}")
                    break

                buf = b""
                while self._running:
                    chunk = self._http_response.read(8192)
                    if not chunk:
                        logger.warning("HTTP MJPEG stream ended, reconnecting...")
                        break
                    buf += chunk

                    # Extract complete JPEG frames from the multipart stream
                    while self._running:
                        # Find boundary
                        boundary_pos = buf.find(boundary)
                        if boundary_pos < 0:
                            break

                        # Find the empty line that separates headers from body
                        header_end = buf.find(b"\r\n\r\n", boundary_pos)
                        if header_end < 0:
                            break  # need more data

                        body_start = header_end + 4

                        # Parse Content-Length from part headers if available
                        header_block = buf[boundary_pos:header_end].decode(
                            "ascii", errors="replace"
                        )
                        content_length = 0
                        for line in header_block.split("\r\n"):
                            if line.lower().startswith("content-length:"):
                                try:
                                    content_length = int(line.split(":", 1)[1].strip())
                                except ValueError:
                                    pass

                        if content_length > 0:
                            # Use Content-Length to extract exact JPEG data
                            if len(buf) < body_start + content_length:
                                break  # need more data
                            jpeg_data = buf[body_start:body_start + content_length]
                            buf = buf[body_start + content_length:]
                        else:
                            # Fallback: scan for JPEG markers (FFD8 start, FFD9 end)
                            jpeg_start = buf.find(b"\xff\xd8", body_start)
                            if jpeg_start < 0:
                                break
                            jpeg_end = buf.find(b"\xff\xd9", jpeg_start + 2)
                            if jpeg_end < 0:
                                break
                            jpeg_data = buf[jpeg_start:jpeg_end + 2]
                            buf = buf[jpeg_end + 2:]

                        # Decode JPEG to PIL Image
                        try:
                            pil_image = Image.open(io.BytesIO(jpeg_data))
                            pil_image.load()  # force decode
                            if pil_image.mode != "RGB":
                                pil_image = pil_image.convert("RGB")
                        except Exception as e:
                            logger.debug(f"Failed to decode JPEG frame: {e}")
                            continue

                        last_inference_push = self._process_frame(
                            pil_image, last_inference_push
                        )

            except Exception as e:
                if not self._running:
                    break
                logger.warning(f"HTTP MJPEG stream error: {e}, reconnecting in 2s...")
                time.sleep(2.0)
