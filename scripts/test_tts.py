"""Test TTS integration: load model with TTS, run inference, save audio.

Usage:
    cd video_chat
    ENABLE_TTS=true python -m scripts.test_tts --image test_files/images/test.jpg
    ENABLE_TTS=true python -m scripts.test_tts --source test_files/videos/test.mp4

Saves output audio to test_output_tts.wav in the current directory.
"""

import argparse
import logging
import sys
import time

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test MiniCPM-o 4.5 TTS integration")
    parser.add_argument("--image", type=str, help="Path to a test image")
    parser.add_argument("--source", type=str, help="Path to a video file (uses first frame)")
    parser.add_argument("--output", type=str, default="test_files/audio/output/tts_test.wav", help="Output WAV path")
    parser.add_argument("--instruction", type=str, default="Describe what you see in one short sentence.", help="Instruction prompt")
    args = parser.parse_args()

    # Verify TTS is enabled
    from app.config import ENABLE_TTS
    if not ENABLE_TTS:
        logger.error("TTS is not enabled. Run with ENABLE_TTS=true")
        sys.exit(1)

    # Load model with TTS
    logger.info("Initializing ModelServer with TTS...")
    t0 = time.time()

    from app.model_server import ModelServer
    server = ModelServer()
    load_time = time.time() - t0
    logger.info(f"Model + TTS loaded in {load_time:.1f}s")

    # Prepare test frame(s)
    frames = []
    if args.image:
        img = Image.open(args.image).convert("RGB")
        frames = [img]
        logger.info(f"Using image: {args.image} ({img.size})")
    elif args.source:
        import cv2
        cap = cv2.VideoCapture(args.source)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            logger.error(f"Failed to read frame from {args.source}")
            sys.exit(1)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames = [img]
        logger.info(f"Using first frame from {args.source} ({img.size})")
    else:
        img = Image.new("RGB", (448, 448), color=(70, 130, 180))
        frames = [img]
        logger.info("Using generated test image")

    # Run inference with audio
    instruction = args.instruction
    logger.info(f"Running TTS inference: '{instruction}'")
    t0 = time.time()

    import torch
    audio_chunks = []
    full_text = ""

    for result in server.infer_with_audio(frames, instruction):
        if result.text:
            full_text += result.text
            print(result.text, end="", flush=True)
        if result.audio is not None:
            audio_chunks.append(result.audio)
        if result.is_last:
            break

    infer_time = time.time() - t0
    print()
    logger.info(f"Inference completed in {infer_time:.1f}s ({len(full_text)} chars)")

    # Save audio
    if audio_chunks:
        import soundfile as sf
        waveform = torch.cat(audio_chunks, dim=-1)
        # Shape is (1, N) — squeeze to (N,)
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        audio_np = waveform.cpu().numpy()
        sf.write(args.output, audio_np, samplerate=24000)
        duration = len(audio_np) / 24000
        logger.info(f"Audio saved to {args.output} ({duration:.1f}s, 24kHz)")
    else:
        logger.warning("No audio chunks received — TTS may not be working")

    # VRAM report
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"VRAM allocated: {allocated:.1f} GB, reserved: {reserved:.1f} GB")


if __name__ == "__main__":
    main()
