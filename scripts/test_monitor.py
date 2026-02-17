"""Standalone test: monitor loop (capture + window + model + orchestrator).

Usage:
    cd video_chat
    conda activate video_chat
    python -m scripts.test_monitor --source test_files/videos/test.mp4
    python -m scripts.test_monitor --source test_files/videos/test.mp4 --cycles 2
    python -m scripts.test_monitor --source 0  # webcam

Loads the model, starts frame capture, runs N inference cycles, prints
streaming output to the terminal. No server or browser needed.
"""

import argparse
import asyncio
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


async def run_test(source, instruction: str, cycles: int):
    from app.frame_capture import FrameCapture
    from app.model_server import ModelServer
    from app.monitor_loop import MonitorLoop
    from app.sliding_window import SlidingWindow

    # Load model
    logger.info("Loading model...")
    t0 = time.monotonic()
    model = ModelServer()
    logger.info(f"Model loaded in {time.monotonic() - t0:.1f}s")

    # Set up pipeline
    window = SlidingWindow()
    capture = FrameCapture(on_frame=window.push)

    # Parse source
    try:
        source = int(source)
    except ValueError:
        pass

    # Start capture and wait for frames to accumulate
    logger.info(f"Starting capture from: {source}")
    capture.start(source)

    # Wait until we have enough frames for the first inference
    from app.config import FRAMES_PER_INFERENCE

    logger.info(f"Waiting for {FRAMES_PER_INFERENCE} frames to accumulate...")
    while window.count < FRAMES_PER_INFERENCE:
        await asyncio.sleep(0.5)
    logger.info(f"Got {window.count} frames, starting monitor loop")

    # Start monitor loop and wait for it to be ready
    monitor = MonitorLoop(model, window)
    loop_task = asyncio.create_task(monitor.run())
    await monitor.wait_started()

    # Set instruction -- this triggers the first cycle immediately
    monitor.set_instruction(instruction)

    # Read streaming output for N cycles
    completed = 0
    t_start = time.monotonic()

    print(f"\n{'='*60}")
    print(f"Instruction: {instruction}")
    print(f"Running {cycles} inference cycle(s)...")
    print(f"{'='*60}\n")

    async for chunk in monitor.stream():
        if chunk is None:
            completed += 1
            elapsed = time.monotonic() - t_start
            print(f"\n\n--- Cycle {completed}/{cycles} complete ({elapsed:.1f}s) ---\n")
            if completed >= cycles:
                break
            t_start = time.monotonic()
        else:
            print(chunk, end="", flush=True)

    # Test instruction change if we have more than 1 cycle
    if cycles > 1:
        t_start = time.monotonic()
        new_instruction = "list only the colors you see, nothing else"
        print(f"\n{'='*60}")
        print(f"Testing instruction change: {new_instruction}")
        print(f"{'='*60}\n")
        monitor.set_instruction(new_instruction)

        async for chunk in monitor.stream():
            if chunk is None:
                elapsed = time.monotonic() - t_start
                print(f"\n\n--- Bonus cycle with new instruction ({elapsed:.1f}s) ---\n")
                break
            else:
                print(chunk, end="", flush=True)

    # Cleanup
    monitor.stop()
    capture.stop()
    await loop_task

    print(f"Total cycles completed: {monitor.cycle_count}")
    print("Test passed.")


def main():
    parser = argparse.ArgumentParser(description="Test monitor loop end-to-end")
    parser.add_argument(
        "--source",
        default="test_files/videos/test.mp4",
        help="Video source: device ID (int) or file path. Default: test video",
    )
    parser.add_argument(
        "--instruction",
        default="describe what you see in these frames",
        help="Instruction to send to the model",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Number of inference cycles to run. Default: 3",
    )
    args = parser.parse_args()
    asyncio.run(run_test(args.source, args.instruction, args.cycles))


if __name__ == "__main__":
    main()
