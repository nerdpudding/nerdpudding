"""Standalone test: load the model and run a single inference.

Usage:
    cd video_chat
    python -m scripts.test_model [--image path/to/image.jpg]

If no image is provided, generates a simple test image.
"""

import argparse
import logging
import sys
import time

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def create_test_image() -> Image.Image:
    """Create a simple colored test image."""
    img = Image.new("RGB", (448, 448), color=(70, 130, 180))
    return img


def main():
    parser = argparse.ArgumentParser(description="Test MiniCPM-o 4.5 model loading and inference")
    parser.add_argument("--image", type=str, help="Path to a test image (optional)")
    args = parser.parse_args()

    # Load model
    logger.info("Initializing ModelServer...")
    t0 = time.time()

    from app.model_server import ModelServer

    server = ModelServer()
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    # Prepare test image
    if args.image:
        img = Image.open(args.image).convert("RGB")
        logger.info(f"Using image: {args.image} ({img.size})")
    else:
        img = create_test_image()
        logger.info(f"Using generated test image ({img.size})")

    # Run inference (streaming)
    logger.info("Running streaming inference...")
    instruction = "Describe what you see in this image in detail."
    t0 = time.time()

    full_response = ""
    for chunk in server.infer([img], instruction, stream=True):
        full_response += chunk
        print(chunk, end="", flush=True)

    infer_time = time.time() - t0
    print()
    logger.info(f"Inference completed in {infer_time:.1f}s ({len(full_response)} chars)")

    # VRAM report
    import torch

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"VRAM allocated: {allocated:.1f} GB, reserved: {reserved:.1f} GB")


if __name__ == "__main__":
    main()
