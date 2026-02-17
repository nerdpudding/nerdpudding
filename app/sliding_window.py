import threading
import time
from collections import deque
from typing import Optional

from PIL import Image

from app.config import WINDOW_SIZE


class FrameMeta:
    """Metadata for a captured frame."""

    __slots__ = ("frame_id", "timestamp", "image")

    def __init__(self, frame_id: int, timestamp: float, image: Image.Image):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.image = image


class SlidingWindow:
    """Thread-safe ring buffer holding the last N frames with metadata.

    The capture thread pushes frames, the inference loop reads them.
    Old frames are auto-evicted by the deque maxlen.
    """

    def __init__(self, max_frames: int = WINDOW_SIZE):
        self._buffer: deque[FrameMeta] = deque(maxlen=max_frames)
        self._lock = threading.Lock()
        self._frame_counter = 0

    def push(self, frame: Image.Image) -> None:
        """Add a frame with an auto-incrementing ID and wall-clock timestamp."""
        with self._lock:
            self._frame_counter += 1
            self._buffer.append(
                FrameMeta(self._frame_counter, time.time(), frame)
            )

    def get_frames(self, n: Optional[int] = None) -> list[Image.Image]:
        """Return the last n frames (images only, backward compatible).

        Args:
            n: Number of frames to return. None returns all available frames.
        """
        with self._lock:
            items = list(self._buffer)
        if n is not None:
            items = items[-n:]
        return [m.image for m in items]

    def get_frames_with_meta(self, n: Optional[int] = None) -> list[FrameMeta]:
        """Return the last n frames with metadata (frame_id, timestamp, image).

        Args:
            n: Number of frames to return. None returns all available frames.
        """
        with self._lock:
            items = list(self._buffer)
        if n is not None:
            items = items[-n:]
        return items

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._frame_counter = 0
