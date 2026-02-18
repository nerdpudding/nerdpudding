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
# Path to the model directory. Default: AWQ INT4 (~8.6 GB VRAM via nvidia-smi).
# For full precision, use MODEL_PATH=models/MiniCPM-o-4_5 (BF16, ~18.5 GB VRAM).
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
# Max frames kept in the ring buffer. At 2 FPS, 32 = 16 seconds of history.
# Must be large enough to hold STREAM_DELAY_INIT worth of frames plus margin.
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "32"))

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
# Default: localhost only. Set to 0.0.0.0 to allow network access (e.g. Docker).
# WARNING: This server has no authentication. Do not expose on untrusted networks.
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8199"))

# JPEG quality for the /api/frame endpoint (1-100). Lower = smaller, faster.
FRAME_JPEG_QUALITY = int(os.getenv("FRAME_JPEG_QUALITY", "80"))

# ---------------------------------------------------------------------------
# MJPEG streaming
# ---------------------------------------------------------------------------
# Display frame rate for the /api/mjpeg endpoint. Controls how often a new
# JPEG is pushed to the browser. At 2 FPS capture, frames repeat between
# captures but the stream stays responsive.
MJPEG_FPS = int(os.getenv("MJPEG_FPS", "10"))

# ---------------------------------------------------------------------------
# Adaptive sync (video-commentary synchronization)
# ---------------------------------------------------------------------------
# Initial delay (seconds) before the first cycle_end calibrates the EMA.
# The MJPEG stream shows frames from this many seconds ago until inference
# timing data is available. Set to 0 to disable delay (real-time, no sync).
STREAM_DELAY_INIT = float(os.getenv("STREAM_DELAY_INIT", "5.0"))

# EMA smoothing factor for adaptive delay. Higher = faster adaptation but
# more jittery. Lower = smoother but slower to respond to changes.
# 0.2 is a good default: adapts within ~5 cycles.
STREAM_DELAY_EMA_ALPHA = float(os.getenv("STREAM_DELAY_EMA_ALPHA", "0.2"))

# ---------------------------------------------------------------------------
# TTS (text-to-speech)
# ---------------------------------------------------------------------------
# Enable TTS audio output. When true, the model generates audio alongside text.
# Adds ~0.6-0.7 GB VRAM for the Token2wav vocoder (float16).
# Requires assets/token2wav/ in the model directory (see docs/model_patches.md).
ENABLE_TTS = os.getenv("ENABLE_TTS", "false").lower() == "true"

# Path to the Token2wav vocoder model files.
# Default: assets/token2wav/ inside the model directory.
TTS_MODEL_DIR = os.getenv("TTS_MODEL_DIR", os.path.join(MODEL_PATH, "assets", "token2wav"))

# Path to the reference audio file for voice cloning.
# Must be a WAV file. Loaded at 16kHz mono.
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", os.path.join(MODEL_PATH, "assets", "HT_ref_audio.wav"))

# Use float16 for the Token2wav vocoder. Saves ~50% VRAM but currently crashes
# with stepaudio2 due to dtype mismatch in flow.setup_cache(). Keep false until fixed.
TTS_FLOAT16 = os.getenv("TTS_FLOAT16", "false").lower() == "true"

# Seconds of silence after TTS audio finishes before the next inference cycle.
# Creates natural breathing room between observations — real commentators pause.
# Only effective when ENABLE_TTS=true. Has no effect in text-only mode.
# 0 = no pause (next cycle starts as soon as audio finishes).
# 1.0 = good default for natural pacing. Increase on faster GPUs if commentary
# feels rushed. Decrease (or set to 0) on slower GPUs where inference time
# already provides enough gap between audio segments.
TTS_PAUSE_AFTER = float(os.getenv("TTS_PAUSE_AFTER", "1.0"))

# Max tokens per inference when TTS is active. Lower = shorter audio output.
# Controls the upper bound on how long each commentary segment can be.
# 150 tokens ≈ 2-3 sentences ≈ 8-12 seconds of audio. Good for natural pacing.
# Only effective when ENABLE_TTS=true. Text-only mode uses MAX_NEW_TOKENS (512).
# Set to 0 to use MAX_NEW_TOKENS for TTS too (not recommended — can produce
# 30+ seconds of audio per cycle, causing queue buildup).
# On slower GPUs, consider lowering to 96-128 for faster cycle times.
# On faster GPUs, 150-200 works well.
TTS_MAX_NEW_TOKENS = int(os.getenv("TTS_MAX_NEW_TOKENS", "150"))
