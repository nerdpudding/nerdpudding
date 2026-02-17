"""Central configuration — single source of truth for all settings.

Every parameter is overridable via environment variable. To change a
setting without editing this file:

    CAPTURE_FPS=2 FRAMES_PER_INFERENCE=4 python -m app.main

Or export them in your shell / .env file.
"""

import os

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
# Which GPU(s) to expose to PyTorch. "0" = first GPU only.
# Set to "0,1" for multi-GPU, or leave empty for CPU-only (not recommended).
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Path to the model directory. Default: AWQ INT4 (~7 GB VRAM).
# For full precision, use MODEL_PATH=models/MiniCPM-o-4_5 (BF16, ~16 GB VRAM).
MODEL_PATH = os.getenv("MODEL_PATH", "models/MiniCPM-o-4_5-awq")

# Token IDs to suppress during generation. 151667 = <think> token.
# Comma-separated list of ints.
SUPPRESS_TOKENS = [
    int(t) for t in os.getenv("SUPPRESS_TOKENS", "151667").split(",")
]

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
# Max tokens the model generates per cycle. Lower = faster responses.
# 512 for detailed descriptions, 128-256 for short commentary.
# Reduce if inference is too slow on your hardware.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

# Image slices per frame. 1 = 64 tokens/frame (fast, lower detail).
# Higher = more detail but more tokens and VRAM per frame.
# If running out of VRAM, keep this at 1.
MAX_SLICE_NUMS = int(os.getenv("MAX_SLICE_NUMS", "1"))

# Max input length in tokens. Limits total prompt + image tokens.
MAX_INP_LENGTH = int(os.getenv("MAX_INP_LENGTH", "4352"))

# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------
# How many frames per second to capture from the video source.
# Higher = more temporal resolution and tighter frame spacing.
# At 2.0 FPS with FRAME_STRIDE=2, inference frames are 1s apart.
CAPTURE_FPS = float(os.getenv("CAPTURE_FPS", "2.0"))

# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------
# Max frames kept in the ring buffer. At 1 FPS, 16 = 16 seconds of history.
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "16"))

# How many recent frames to send per inference cycle.
# More frames = more temporal context but slower inference and more VRAM.
# Token cost per cycle = FRAMES_PER_INFERENCE * 64 * MAX_SLICE_NUMS.
FRAMES_PER_INFERENCE = int(os.getenv("FRAMES_PER_INFERENCE", "4"))

# Frame stride: take every Nth frame from the buffer instead of consecutive.
# With CAPTURE_FPS=2 and FRAME_STRIDE=2, 4 frames span 4 seconds (every other
# frame at 0.5s intervals). Stride=1 means consecutive (no skipping).
# Higher stride = more temporal coverage per frame set, at the cost of
# missing short-lived events between sampled frames.
FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", "2"))

# ---------------------------------------------------------------------------
# Monitor loop
# ---------------------------------------------------------------------------
# Seconds between inference cycles. Keep low to minimize idle time between
# cycles. The actual cycle rate is limited by inference speed (~2s), so
# this is just the minimum pause. Set higher for less frequent commentary.
INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "1.0"))

# Mean pixel difference threshold (0-255) for scene change detection.
# Below this = scene unchanged, cycle is skipped. 0 = never skip.
# 5.0 works for animation. May need tuning for live video (lower)
# or mostly static scenes (higher). Set to 0 to never skip.
CHANGE_THRESHOLD = float(os.getenv("CHANGE_THRESHOLD", "5.0"))

# System prompt prepended to every inference call. Shapes the model's
# response style. Override with a custom prompt for different use cases.
COMMENTATOR_PROMPT = os.getenv("COMMENTATOR_PROMPT", (
    "You are a live video commentator watching a continuous stream. "
    "Rules:\n"
    "- Keep each response to 1-2 short sentences.\n"
    "- Only mention what is NEW or has CHANGED.\n"
    "- Do NOT repeat or rephrase your previous comment.\n"
    "- If nothing notable changed, respond with exactly '...' and nothing else.\n"
    "- Be specific: mention objects, actions, colors, text on screen."
))

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
# Host and port for the FastAPI server.
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8199"))

# JPEG quality for the /api/frame endpoint (1-100). Lower = smaller, faster.
FRAME_JPEG_QUALITY = int(os.getenv("FRAME_JPEG_QUALITY", "80"))
