"""Audio delivery: resample TTS output and distribute to consumers.

Receives 24kHz float32 audio chunks from the model's TTS, resamples
to 48kHz int16 PCM (WebRTC / playback standard), and publishes to
subscribers via async queues (same pub/sub pattern as MonitorLoop).
"""

import asyncio
import logging
from typing import Optional

import numpy as np
import torch
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


class AudioManager:
    """Receives TTS audio, resamples, and delivers to consumers."""

    def __init__(self):
        self._subscribers: set[asyncio.Queue[Optional[bytes]]] = set()

    def subscribe(self) -> asyncio.Queue[Optional[bytes]]:
        """Subscribe to audio events. Returns a queue that receives:

        - bytes: PCM audio chunk (48kHz, mono, int16 LE)
        - None: stop signal
        """
        q: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._subscribers.add(q)
        logger.debug(f"Audio subscriber added (total: {len(self._subscribers)})")
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        self._subscribers.discard(q)
        logger.debug(f"Audio subscriber removed (total: {len(self._subscribers)})")

    def publish(self, data: Optional[bytes]) -> None:
        """Send audio data to all subscribers. Call from event loop thread."""
        for q in self._subscribers:
            q.put_nowait(data)

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
