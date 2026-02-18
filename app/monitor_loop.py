import asyncio
import logging
import time
from typing import AsyncGenerator, Optional, Union

import numpy as np
from PIL import Image

from app.audio_manager import AudioManager
from app.config import (
    CHANGE_THRESHOLD,
    COMMENTATOR_PROMPT,
    ENABLE_TTS,
    FRAME_STRIDE,
    FRAMES_PER_INFERENCE,
    INFERENCE_INTERVAL,
    STREAM_DELAY_EMA_ALPHA,
    STREAM_DELAY_INIT,
    TTS_PAUSE_AFTER,
)
from app.model_server import ModelServer
from app.sliding_window import SlidingWindow

logger = logging.getLogger(__name__)


class MonitorLoop:
    """Orchestrator: periodically infers on recent frames and streams output.

    Two modes:
    - IDLE: frames buffered, no inference (no instruction set)
    - ACTIVE: periodic inference with current instruction

    Uses pub/sub for output: multiple consumers (SSE, WebSocket, test scripts)
    can each subscribe and independently receive all events.
    """

    def __init__(self, model: ModelServer, window: SlidingWindow,
                 audio_manager: Optional[AudioManager] = None):
        self._model = model
        self._window = window
        self._audio_manager = audio_manager
        self._instruction: Optional[str] = None
        self._commentator_prompt: str = COMMENTATOR_PROMPT
        self._running = False
        self._generating = False
        self._stop_requested = False
        self._started = asyncio.Event()
        self._cycle_event = asyncio.Event()
        # Events: str = text chunk, dict = cycle metadata, None = stop
        self._subscribers: set[asyncio.Queue[Union[str, dict, None]]] = set()
        self._cycle_count = 0
        self._last_response: str = ""
        self._last_instruction: Optional[str] = None
        self._last_inference_frame: Optional[Image.Image] = None
        # Adaptive sync: EMA-smoothed delay for MJPEG stream
        self._target_delay: float = STREAM_DELAY_INIT

    @property
    def mode(self) -> str:
        return "ACTIVE" if self._instruction else "IDLE"

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_generating(self) -> bool:
        return self._generating

    @property
    def instruction(self) -> Optional[str]:
        return self._instruction

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def target_delay(self) -> float:
        """Current adaptive delay for MJPEG sync (seconds)."""
        return self._target_delay

    def set_instruction(self, instruction: Optional[str]) -> None:
        """Set or clear the current instruction.

        Setting a new instruction triggers an immediate inference cycle
        if the model is not currently generating. Also resets context
        carry-over since the focus changed.
        """
        old = self._instruction
        self._instruction = instruction
        if instruction and instruction != old:
            self._last_response = ""
            self._last_inference_frame = None
            if not self._generating:
                self._cycle_event.set()
        logger.info(f"Instruction {'set' if instruction else 'cleared'}: {instruction}")

    def set_commentator_prompt(self, prompt: str) -> None:
        """Switch the system prompt (e.g. when user selects a different profile)."""
        self._commentator_prompt = prompt
        self._last_response = ""
        self._last_inference_frame = None
        logger.info(f"Commentator prompt changed ({len(prompt)} chars)")

    def subscribe(self) -> asyncio.Queue[Union[str, dict, None]]:
        """Subscribe to output events. Returns a queue that receives:

        - str: text chunk from model
        - dict: cycle metadata (type="cycle_end", timing, frame IDs)
        - None: stop signal
        """
        q: asyncio.Queue[Union[str, dict, None]] = asyncio.Queue(maxsize=500)
        self._subscribers.add(q)
        logger.debug(f"Subscriber added (total: {len(self._subscribers)})")
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        self._subscribers.discard(q)
        logger.debug(f"Subscriber removed (total: {len(self._subscribers)})")

    def _publish(self, item) -> None:
        """Send an item to all subscribers. Called from the event loop thread."""
        for q in self._subscribers:
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                pass  # Drop data for slow consumers

    def _scene_diff(self, current_frame: Image.Image) -> float:
        """Compute mean pixel difference from last inference frame.

        Returns 255.0 if no previous frame (first cycle).
        Returns float in range 0-255.
        """
        if self._last_inference_frame is None:
            return 255.0
        try:
            old = np.array(self._last_inference_frame.resize((64, 64)), dtype=np.float32)
            new = np.array(current_frame.resize((64, 64)), dtype=np.float32)
            diff = float(np.mean(np.abs(old - new)))
            logger.debug(f"Scene diff: {diff:.1f} (threshold: {CHANGE_THRESHOLD})")
            return diff
        except Exception:
            return 255.0

    def _commentary_intensity(self, scene_diff: float) -> str:
        """Determine commentary length hint based on scene diff and recent history.

        Uses two signals:
        1. Pixel diff (visual change) — fast, no extra inference cost
        2. Previous response length (semantic signal) — if the model's last
           response was short, it saw nothing interesting

        Returns: 'minimal', 'brief', or 'normal'.
        """
        prev_len = len(self._last_response.strip()) if self._last_response else 0
        prev_was_short = prev_len < 40  # "..." is 3, a short phrase is ~20-40 chars

        if scene_diff < 15 and prev_was_short:
            return "minimal"
        elif scene_diff < 15 or (scene_diff < 40 and prev_was_short):
            return "brief"
        else:
            return "normal"

    def _build_prompt(self, instruction: str, scene_diff: float = 255.0) -> str:
        """Build the full prompt with commentator system message and context."""
        parts = [self._commentator_prompt]
        if self._last_response and self._last_response.strip() != "...":
            parts.append(
                f'\nYour last comment was: "{self._last_response}"\n'
                "Do not repeat this. Only add new observations."
            )
        if self._model.tts_enabled:
            intensity = self._commentary_intensity(scene_diff)
            if intensity == "minimal":
                parts.append("\nVery little changed. Be extremely brief — one short phrase at most.")
            elif intensity == "brief":
                parts.append("\nSome things changed. Keep it to one sentence.")
        parts.append(f"\nFocus: {instruction}")
        return "\n".join(parts)

    async def wait_started(self) -> None:
        """Wait until run() is actively looping. Use before set_instruction/stream."""
        await self._started.wait()

    async def run(self) -> None:
        """Main loop. Call from an async context (e.g. asyncio.run or FastAPI startup)."""
        if self._stop_requested:
            return
        self._running = True
        self._started.set()
        logger.info("Monitor loop started")

        while self._running:
            try:
                await asyncio.wait_for(
                    self._cycle_event.wait(),
                    timeout=INFERENCE_INTERVAL,
                )
            except asyncio.TimeoutError:
                pass
            self._cycle_event.clear()

            if not self._running:
                break

            if not self._instruction:
                continue

            if self._generating:
                continue

            frame_metas = self._window.get_frames_with_meta(FRAMES_PER_INFERENCE, stride=FRAME_STRIDE)
            if not frame_metas:
                logger.debug("No frames available, skipping cycle")
                continue

            # Change detection: skip if scene hasn't changed enough
            newest_frame = frame_metas[-1].image
            instruction_changed = self._instruction != self._last_instruction
            scene_diff = self._scene_diff(newest_frame)
            if not instruction_changed and scene_diff < CHANGE_THRESHOLD:
                logger.info(f"Scene unchanged (diff={scene_diff:.1f}), skipping cycle")
                continue

            self._last_instruction = self._instruction
            await self._run_cycle(frame_metas, self._instruction, scene_diff)

            # Audio gate: wait for browser to finish playing + breathing pause
            if self._audio_manager is not None:
                playback_end = self._audio_manager.estimated_playback_end
                remaining = playback_end - time.time() + TTS_PAUSE_AFTER
                if remaining > 0:
                    logger.info(f"Audio gate: waiting {remaining:.1f}s (playback + pause)")
                    await asyncio.sleep(remaining)

        self._running = False
        self._started.clear()
        logger.info("Monitor loop stopped")

    async def _run_cycle(self, frame_metas: list, instruction: str,
                         scene_diff: float = 255.0) -> None:
        """Run one inference cycle in a thread pool."""
        self._generating = True
        self._cycle_count += 1
        if self._audio_manager is not None:
            self._audio_manager.reset_clock()
        cycle_num = self._cycle_count
        frame_ids = [m.frame_id for m in frame_metas]
        frame_timestamps = [m.timestamp for m in frame_metas]
        images = [m.image for m in frame_metas]
        prompt = self._build_prompt(instruction, scene_diff)
        label = instruction[:50] + "..." if len(instruction) > 50 else instruction
        logger.info(f"Cycle {cycle_num}: {len(images)} frames (#{frame_ids[0]}-#{frame_ids[-1]}), instruction='{label}'")

        t0 = time.time()
        try:
            loop = asyncio.get_running_loop()
            full_response = await loop.run_in_executor(
                None,
                self._inference_worker,
                images,
                prompt,
                loop,
                cycle_num,
                frame_ids,
                frame_timestamps,
                t0,
            )
            self._last_response = full_response.strip()
            self._last_inference_frame = images[-1]
        except Exception:
            logger.exception(f"Cycle {cycle_num} failed")
        finally:
            elapsed = time.time() - t0
            logger.info(f"Cycle {cycle_num} done in {elapsed:.1f}s")
            self._generating = False

    def _inference_worker(self, frames, prompt, loop,
                          cycle_num, frame_ids, frame_timestamps, t0) -> str:
        """Runs in thread pool. Streams chunks to all subscribers. Returns full response."""
        chunks = []
        if self._model.tts_enabled:
            # Buffer audio until we know the response is not "..." (skip signal).
            # The Token2wav vocoder produces Chinese speech artifacts on "...",
            # so we suppress audio for skip responses entirely.
            audio_buffer = []
            streaming_audio = False
            for result in self._model.infer_with_audio(frames, prompt):
                if result.text:
                    chunks.append(result.text)
                    loop.call_soon_threadsafe(self._publish, result.text)
                    # Once accumulated text exceeds skip signal, flush audio buffer
                    if not streaming_audio and len("".join(chunks).strip()) > 5:
                        streaming_audio = True
                        if self._audio_manager is not None:
                            for buffered in audio_buffer:
                                pcm = AudioManager.resample_to_48k_int16(buffered)
                                loop.call_soon_threadsafe(self._audio_manager.publish, pcm)
                        audio_buffer.clear()
                if result.audio is not None:
                    if streaming_audio and self._audio_manager is not None:
                        pcm = AudioManager.resample_to_48k_int16(result.audio)
                        loop.call_soon_threadsafe(self._audio_manager.publish, pcm)
                    elif not streaming_audio:
                        audio_buffer.append(result.audio)
                if result.is_last:
                    break
            # Flush remaining buffer if response was real (not "...")
            full_text = "".join(chunks).strip()
            if full_text != "..." and audio_buffer and self._audio_manager is not None:
                for buffered in audio_buffer:
                    pcm = AudioManager.resample_to_48k_int16(buffered)
                    loop.call_soon_threadsafe(self._audio_manager.publish, pcm)
            if full_text == "...":
                logger.debug("Skip response '...' — audio suppressed")
        else:
            for chunk in self._model.infer(frames, prompt, stream=True):
                chunks.append(chunk)
                loop.call_soon_threadsafe(self._publish, chunk)
        full_response = "".join(chunks)
        t_end = time.time()
        meta = {
            "type": "cycle_end",
            "cycle": cycle_num,
            "frame_ids": frame_ids,
            "oldest_frame_at": frame_timestamps[0],
            "newest_frame_at": frame_timestamps[-1],
            "inference_start": t0,
            "inference_end": t_end,
            "inference_sec": round(t_end - t0, 2),
            "latency_sec": round(t_end - frame_timestamps[0], 2),
            "skipped": full_response.strip() == "...",
        }
        # Update adaptive delay via EMA on observed latency.
        # Skip "..." responses — they have artificially low latency that
        # would pull the EMA down and desync real commentary cycles.
        if STREAM_DELAY_INIT > 0 and not meta["skipped"]:
            observed = t_end - frame_timestamps[-1]
            alpha = STREAM_DELAY_EMA_ALPHA
            old_delay = self._target_delay
            self._target_delay = (1 - alpha) * old_delay + alpha * observed
            logger.info(
                f"Sync delay: observed={observed:.2f}s, "
                f"target={old_delay:.2f}s -> {self._target_delay:.2f}s"
            )
        if STREAM_DELAY_INIT > 0:
            meta["target_delay"] = round(self._target_delay, 2)

        loop.call_soon_threadsafe(self._publish, meta)
        return full_response

    async def stream(self) -> AsyncGenerator[Union[str, dict, None], None]:
        """Yield events as they arrive.

        - str: text chunk from model
        - dict: cycle metadata (type="cycle_end")
        - None: stop signal (terminates the generator)

        Each call creates an independent subscriber -- safe for multiple
        concurrent consumers (SSE connections, WebSocket clients, etc.).
        """
        q = self.subscribe()
        try:
            while self._running or not q.empty():
                try:
                    event = await asyncio.wait_for(q.get(), timeout=1.0)
                    if event is None:
                        return
                    yield event
                except asyncio.TimeoutError:
                    continue
        finally:
            self.unsubscribe(q)

    def stop(self) -> None:
        """Stop the monitor loop. Must be called from the async context."""
        self._stop_requested = True
        self._running = False
        self._cycle_event.set()
        self._publish(None)
