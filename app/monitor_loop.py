import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

from app.config import FRAMES_PER_INFERENCE, INFERENCE_INTERVAL
from app.model_server import ModelServer
from app.sliding_window import SlidingWindow

logger = logging.getLogger(__name__)


class MonitorLoop:
    """Orchestrator: periodically infers on recent frames and streams output.

    Two modes:
    - IDLE: frames buffered, no inference (no instruction set)
    - ACTIVE: periodic inference with current instruction
    """

    def __init__(self, model: ModelServer, window: SlidingWindow):
        self._model = model
        self._window = window
        self._instruction: Optional[str] = None
        self._running = False
        self._generating = False
        self._stop_requested = False
        self._started = asyncio.Event()
        self._cycle_event = asyncio.Event()
        self._chunks: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._cycle_count = 0

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

    def set_instruction(self, instruction: Optional[str]) -> None:
        """Set or clear the current instruction.

        Setting a new instruction triggers an immediate inference cycle
        if the model is not currently generating.
        """
        old = self._instruction
        self._instruction = instruction
        if instruction and instruction != old and not self._generating:
            self._cycle_event.set()
        logger.info(f"Instruction {'set' if instruction else 'cleared'}: {instruction}")

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

            frames = self._window.get_frames(FRAMES_PER_INFERENCE)
            if not frames:
                logger.debug("No frames available, skipping cycle")
                continue

            await self._run_cycle(frames, self._instruction)

        self._running = False
        self._started.clear()
        logger.info("Monitor loop stopped")

    async def _run_cycle(self, frames: list, instruction: str) -> None:
        """Run one inference cycle in a thread pool."""
        self._generating = True
        self._cycle_count += 1
        cycle_num = self._cycle_count
        n_frames = len(frames)
        label = instruction[:50] + "..." if len(instruction) > 50 else instruction
        logger.info(f"Cycle {cycle_num}: {n_frames} frames, instruction='{label}'")

        t0 = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._inference_worker,
                frames,
                instruction,
                loop,
            )
        except Exception:
            logger.exception(f"Cycle {cycle_num} failed")
        finally:
            elapsed = time.monotonic() - t0
            logger.info(f"Cycle {cycle_num} done in {elapsed:.1f}s")
            self._generating = False

    def _inference_worker(self, frames, instruction, loop) -> None:
        """Runs in thread pool. Streams chunks to the async queue."""
        for chunk in self._model.infer(frames, instruction, stream=True):
            loop.call_soon_threadsafe(self._chunks.put_nowait, chunk)
        loop.call_soon_threadsafe(self._chunks.put_nowait, None)

    async def stream(self) -> AsyncGenerator[Optional[str], None]:
        """Yield text chunks as they arrive. None marks end of an inference cycle."""
        while self._running or not self._chunks.empty():
            try:
                chunk = await asyncio.wait_for(self._chunks.get(), timeout=1.0)
                yield chunk
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """Stop the monitor loop."""
        self._stop_requested = True
        self._running = False
        self._cycle_event.set()
        self._chunks.put_nowait(None)
