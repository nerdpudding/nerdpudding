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

    def get_frames(
        self, n: Optional[int] = None, stride: int = 1
    ) -> list[Image.Image]:
        """Return the last n frames (images only, backward compatible).

        Args:
            n: Number of frames to return. None returns all available frames.
            stride: Take every Nth frame from the tail. stride=2 with n=4
                    returns 4 frames spanning 8 buffer positions.
        """
        return [m.image for m in self.get_frames_with_meta(n, stride)]

    def get_frames_with_meta(
        self, n: Optional[int] = None, stride: int = 1
    ) -> list[FrameMeta]:
        """Return the last n frames with metadata (frame_id, timestamp, image).

        Args:
            n: Number of frames to return. None returns all available frames.
            stride: Take every Nth frame from the tail of the buffer.
                    stride=1 means consecutive (default, backward compatible).
                    stride=2 means every other frame, etc.
                    If fewer frames are available than n*stride, returns what's
                    available with the given stride.
        """
        with self._lock:
            items = list(self._buffer)
        if stride > 1:
            # Take every stride-th frame from the end
            items = items[::-1][::stride][::-1]
        if n is not None:
            items = items[-n:]
        return items

    def get_frame_near(self, target_time: float) -> Optional[FrameMeta]:
        """Return the frame closest to target_time, or None if buffer is empty.

        Uses binary-style search on the timestamp-sorted buffer. Since frames
        are appended in chronological order, the deque is always sorted.
        """
        with self._lock:
            if not self._buffer:
                return None
            # Edge cases: target before/after all frames
            if target_time <= self._buffer[0].timestamp:
                return self._buffer[0]
            if target_time >= self._buffer[-1].timestamp:
                return self._buffer[-1]
            # Linear scan (buffer is small, max ~32 items)
            best = self._buffer[0]
            best_diff = abs(best.timestamp - target_time)
            for meta in self._buffer:
                diff = abs(meta.timestamp - target_time)
                if diff < best_diff:
                    best = meta
                    best_diff = diff
            return best

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._frame_counter = 0
