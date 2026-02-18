"""Audio delivery: resample TTS output and distribute to consumers.

Receives 24kHz float32 audio chunks from the model's TTS, resamples
to 48kHz int16 PCM (WebRTC / playback standard), and publishes to
subscribers via async queues (same pub/sub pattern as MonitorLoop).
"""

import asyncio
import logging
import time
from typing import Optional

import numpy as np
import torch
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


class AudioManager:
    """Receives TTS audio, resamples, and delivers to consumers.

    Also tracks an audio clock to estimate when the browser finishes
    playing the current cycle's audio. Used by MonitorLoop to gate
    the next inference cycle (prevents queue buildup / drift).
    """

    # Output format after resample_to_48k_int16()
    SAMPLE_RATE = 48000
    BYTES_PER_SAMPLE = 2  # int16

    def __init__(self):
        self._subscribers: set[asyncio.Queue[Optional[bytes]]] = set()
        # Audio clock — tracks playback timing per cycle
        self._first_publish_time: Optional[float] = None
        self._audio_seconds: float = 0.0

    def subscribe(self) -> asyncio.Queue[Optional[bytes]]:
        """Subscribe to audio events. Returns a queue that receives:

        - bytes: PCM audio chunk (48kHz, mono, int16 LE)
        - None: stop signal
        """
        q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=200)
        self._subscribers.add(q)
        logger.debug(f"Audio subscriber added (total: {len(self._subscribers)})")
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        self._subscribers.discard(q)
        logger.debug(f"Audio subscriber removed (total: {len(self._subscribers)})")

    def publish(self, data: Optional[bytes]) -> None:
        """Send audio data to all subscribers. Also tracks audio clock.

        NOTE: data is ALREADY resampled 48kHz int16 PCM (via
        resample_to_48k_int16() in monitor_loop._inference_worker).
        So len(data) / (SAMPLE_RATE * BYTES_PER_SAMPLE) = seconds of audio.
        """
        if data is not None:
            if self._first_publish_time is None:
                self._first_publish_time = time.time()
            self._audio_seconds += len(data) / (self.SAMPLE_RATE * self.BYTES_PER_SAMPLE)
        for q in self._subscribers:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass  # Drop data for slow consumers

    def reset_clock(self) -> None:
        """Reset audio clock for a new cycle. Call at cycle start."""
        self._first_publish_time = None
        self._audio_seconds = 0.0

    @property
    def estimated_playback_end(self) -> float:
        """Wall-clock time when the browser is estimated to finish playing.

        Returns 0.0 if no audio was published this cycle (no waiting needed).
        """
        if self._first_publish_time is None:
            return 0.0
        return self._first_publish_time + self._audio_seconds

    def stop(self) -> None:
        """Send stop signal to all subscribers."""
        self.publish(None)

    @staticmethod
    def resample_to_48k_int16(audio_24k: torch.Tensor) -> bytes:
        """Convert a 24kHz float32 tensor to 48kHz int16 PCM bytes.

        Args:
            audio_24k: Tensor of shape (1, N) or (N,), float32, 24kHz.

        Returns:
            Raw PCM bytes: 48kHz, mono, int16 little-endian.
        """
        audio_np = audio_24k.cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze(0)

        # Resample 24kHz -> 48kHz (exact factor of 2)
        audio_48k = resample_poly(audio_np, up=2, down=1).astype(np.float32)

        # float32 [-1, 1] -> int16 [-32768, 32767]
        audio_int16 = np.clip(audio_48k * 32767, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()
