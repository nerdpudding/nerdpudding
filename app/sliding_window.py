import threading
import time
from collections import deque
from typing import Optional

from PIL import Image

from app.config import WINDOW_SIZE


class SlidingWindow:
    """Thread-safe ring buffer holding the last N frames with timestamps.

    The capture thread pushes frames, the inference loop reads them.
    Old frames are auto-evicted by the deque maxlen.
    """

    def __init__(self, max_frames: int = WINDOW_SIZE):
        self._buffer: deque[tuple[float, Image.Image]] = deque(maxlen=max_frames)
        self._lock = threading.Lock()

    def push(self, frame: Image.Image) -> None:
        """Add a frame with the current timestamp."""
        with self._lock:
            self._buffer.append((time.monotonic(), frame))

    def get_frames(self, n: Optional[int] = None) -> list[Image.Image]:
        """Return the last n frames (without timestamps).

        Args:
            n: Number of frames to return. None returns all available frames.
        """
        with self._lock:
            items = list(self._buffer)

        if n is not None:
            items = items[-n:]

        return [frame for _, frame in items]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
