"""Central configuration — single source of truth for all settings.

Every parameter is overridable via environment variable:
    CAPTURE_FPS=5 ENABLE_TTS=true python -m app.main

Or export them in your shell / .env file. Server restart required for changes.

# ===== ACTIVE (update when you switch!) =====================================
#   Section 2: Push it   |   Section 3: Preset B   →   Combo: "Sniper"
#
# ===== COMBOS (Section 2 + Section 3) ======================================
#   Combo          GPU preset       Tune preset      Use case
#   Beast          Push it     +    D: Beast         max everything, 4090 full power
#   Sniper         Push it     +    B: Sniper        fast action, webcam, sports
#   Sniper Lite    Conservative +   B: Sniper        fast action, less VRAM
#   Owl            Push it     +    C: Owl           broad context, meetings
#   Sentry         Conservative +   A: Sentry        slow scenes, security cam
#
#   Too slow? (Inference >5s in web UI)  → go one row down
#   OOM crash?                           → switch Section 2 to Conservative
#
# ===== SECTIONS =============================================================
#   1. GPU & Model       — DO NOT TOUCH
#   2. Inference limits   — Ctrl+/ to switch GPU presets
#   3. Tuning presets     — Ctrl+/ to switch A/B/C/D
#   4. Prompt profiles    — switch live from web UI
#   5. Server & streaming — DO NOT TOUCH
#   6. TTS paths          — DO NOT TOUCH
"""

import os

# ===========================================================================
# 1. GPU & MODEL — DO NOT TOUCH unless you switch GPU or model
# ===========================================================================

CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# AWQ INT4 (default): ~8.6 GB VRAM base, ~14-15 GB during inference.
# BF16 full precision: ~18.5 GB base. Only if you have 24+ GB free.
MODEL_PATH = os.getenv("MODEL_PATH", "models/MiniCPM-o-4_5-awq")

# Suppresses the model's internal <think> token. Do not change.
SUPPRESS_TOKENS = [
    int(t) for t in os.getenv("SUPPRESS_TOKENS", "151667").split(",")
]

# ===========================================================================
# 2. INFERENCE LIMITS — how much the AI sees and says per frame
# ===========================================================================
# These 3 settings are tightly linked. Switch them as a group (like presets).
#
# WHAT EACH SETTING DOES:
#
#   MAX_SLICE_NUMS   Detail level per frame. The AI divides each frame into
#                    slices and converts each to 64 tokens.
#                      1 = 64 tokens/frame. Sees shapes, people, large text.
#                          Misses: small text, distant faces, subtle details.
#                      2 = 128 tokens/frame. Sees more detail per frame.
#                          Misses less, but inference is slower per cycle.
#                      3 = 192 tokens/frame. Maximum detail. Diminishing
#                          returns vs 2, and noticeably slower.
#
#   MAX_INP_LENGTH   Hard cap on total input tokens (all image tokens + prompt).
#                    Must fit: FRAMES_PER_INFERENCE * 64 * MAX_SLICE_NUMS + ~200.
#                    If this is too low, frames get silently dropped!
#                    Too high is harmless — it's just an upper bound.
#
#   MAX_NEW_TOKENS   Max response length (text-only mode, without TTS).
#                    With TTS enabled, TTS_MAX_NEW_TOKENS (section 3) overrides
#                    this. So if you use TTS, this setting has no effect.
#                    512 is fine for text-only. No reason to change it.
#
# To switch: uncomment ONE block, comment the others. Restart server.
#
# ---- GPU PRESET: Conservative (RTX 3080/3090, 10-12 GB VRAM) ----
# Low detail, fast inference. For GPUs with limited headroom.
# With Preset D (10 frames): 10 * 64 = 640 image tokens/cycle.
# MAX_SLICE_NUMS = int(os.getenv("MAX_SLICE_NUMS", "1"))
# MAX_INP_LENGTH = int(os.getenv("MAX_INP_LENGTH", "4352"))
# MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

# ---- GPU PRESET: Push it (RTX 4090, 24 GB VRAM) (ACTIVE) ----
# More detail per frame. Your 4090 has ~10 GB headroom — use it.
# With Preset D (10 frames): 10 * 128 = 1280 image tokens/cycle.
# The AI will notice smaller details: text on screens, facial expressions,
# objects in the background that it misses at SLICE=1.
# Tradeoff: inference ~30-50% slower per cycle. Watch cycle times.
MAX_SLICE_NUMS = int(os.getenv("MAX_SLICE_NUMS", "2"))
MAX_INP_LENGTH = int(os.getenv("MAX_INP_LENGTH", "8192"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

# ---- GPU PRESET: Maximum detail (RTX 4090, if SLICE=2 feels insufficient) ----
# Highest detail. Only try if SLICE=2 still misses things you want it to see.
# With Preset D (10 frames): 10 * 192 = 1920 image tokens/cycle.
# Tradeoff: slowest inference. Diminishing returns vs SLICE=2.
# If cycle time gets too long (>6s), go back to SLICE=2.
# MAX_SLICE_NUMS = int(os.getenv("MAX_SLICE_NUMS", "3"))
# MAX_INP_LENGTH = int(os.getenv("MAX_INP_LENGTH", "8192"))
# MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

# ===========================================================================
# 3. TUNING PRESETS — frame capture, pacing, responsiveness
# ===========================================================================
# These settings control HOW the AI observes and HOW FAST it responds.
# They all work together — switching one without the others may not help.
#
# To switch: uncomment ONE preset block, comment the others. Restart server.
# Or test via env vars without editing:
#   CAPTURE_FPS=5 CHANGE_THRESHOLD=2.0 TTS_MAX_NEW_TOKENS=96 python -m app.main
#
# WHAT EACH SETTING DOES:
#
#   CAPTURE_FPS          How often a frame is grabbed for the AI (not display).
#                        Higher = catches faster movements, but fills buffer faster.
#                        Display always runs at full source FPS regardless.
#
#   FRAMES_PER_INFERENCE How many frames the AI sees per inference cycle.
#                        More = more context, but more image tokens = slower.
#
#   FRAME_STRIDE         Take every Nth frame from the buffer.
#                        1 = consecutive frames (dense). 2 = skip every other.
#                        Higher = wider time span but gaps between frames.
#
#   WINDOW_SIZE          Ring buffer capacity. Must be >= FRAMES_PER_INFERENCE
#                        * FRAME_STRIDE + some margin for the adaptive sync.
#
#   CHANGE_THRESHOLD     Mean pixel difference threshold to detect scene change.
#                        Theoretical range 0-255, practical range 0-50.
#                        Below this = "nothing changed", cycle skipped.
#                        5.0 = only reacts to obvious changes (misses subtle ones).
#                        2.0 = reacts to small movements (more cycles, more responsive).
#                        0   = never skip (every cycle runs, most responsive).
#
#   INFERENCE_INTERVAL   Minimum seconds between inference cycles.
#                        Actual rate is limited by inference speed (can't go faster).
#                        0.5 = try a new cycle every 0.5s. 1.0 = every 1s.
#
#   TTS_MAX_NEW_TOKENS   Max output tokens when TTS is on. Controls audio length.
#                        96  = ~4-6 seconds of speech (snappy, fast cycles)
#                        120 = ~6-8 seconds (medium)
#                        150 = ~8-12 seconds (detailed, slower cycles)
#
#   TTS_PAUSE_AFTER      Silence (seconds) after TTS finishes before next cycle.
#                        Creates breathing room. 0 = no pause. 1.0 = natural pace.
#
# KEY FORMULAS:
#   Time window  = FRAMES_PER_INFERENCE * FRAME_STRIDE / CAPTURE_FPS
#   Image tokens = FRAMES_PER_INFERENCE * 64 * MAX_SLICE_NUMS
#
# HOW TO TEST:
#   1. Mix of sitting still AND quick movements (grab something, wave, stand up)
#   2. Check web UI cycle metadata: "Inference: Xs" and "Latency: Xs"
#   3. AI missed your action? → CAPTURE_FPS too low or CHANGE_THRESHOLD too high
#   4. Commentary too slow? → reduce TTS_MAX_NEW_TOKENS or TTS_PAUSE_AFTER
#   5. Commentary too rushed? → increase TTS_PAUSE_AFTER
#
# ---- PRESET A: Sentry ----
# Best for: slow-changing scenes, security cams, nature, saving compute.
# Time window: 4s | Captures every 500ms
# Tokens/cycle: 256 (SLICE=1) or 512 (SLICE=2)
# High threshold: only reacts to big changes (person entering, car passing).
# CAPTURE_FPS = float(os.getenv("CAPTURE_FPS", "2.0"))
# FRAMES_PER_INFERENCE = int(os.getenv("FRAMES_PER_INFERENCE", "4"))
# FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", "2"))
# WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "32"))
# CHANGE_THRESHOLD = float(os.getenv("CHANGE_THRESHOLD", "5.0"))
# INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "1.0"))
# TTS_MAX_NEW_TOKENS = int(os.getenv("TTS_MAX_NEW_TOKENS", "150"))
# TTS_PAUSE_AFTER = float(os.getenv("TTS_PAUSE_AFTER", "1.0"))

# ---- PRESET B: Sniper (ACTIVE) ----
# Best for: webcam, sports, anything with quick movements.
# Time window: 0.8s | Captures every 200ms | 512 image tokens/cycle
# No threshold: every cycle runs, never skips.
CAPTURE_FPS = float(os.getenv("CAPTURE_FPS", "5.0"))
FRAMES_PER_INFERENCE = int(os.getenv("FRAMES_PER_INFERENCE", "4"))
FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", "1"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "48"))
CHANGE_THRESHOLD = float(os.getenv("CHANGE_THRESHOLD", "0"))
INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "0.5"))
TTS_MAX_NEW_TOKENS = int(os.getenv("TTS_MAX_NEW_TOKENS", "96"))
TTS_PAUSE_AFTER = float(os.getenv("TTS_PAUSE_AFTER", "0.5"))

# ---- PRESET C: Owl ----
# Best for: general use — catches fast movements AND has broad context.
# Time window: 4s | Captures every 250ms
# Tokens/cycle: 512 (SLICE=1) or 1024 (SLICE=2)
# Low threshold: reacts to most changes.
# CAPTURE_FPS = float(os.getenv("CAPTURE_FPS", "4.0"))
# FRAMES_PER_INFERENCE = int(os.getenv("FRAMES_PER_INFERENCE", "8"))
# FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", "2"))
# WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "48"))
# CHANGE_THRESHOLD = float(os.getenv("CHANGE_THRESHOLD", "1.0"))
# INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "0.5"))
# TTS_MAX_NEW_TOKENS = int(os.getenv("TTS_MAX_NEW_TOKENS", "120"))
# TTS_PAUSE_AFTER = float(os.getenv("TTS_PAUSE_AFTER", "0.5"))

# ---- PRESET D: Beast ----
# Best for: RTX 4090 with VRAM to spare, want the AI to see everything.
# Time window: 4s | Captures every 200ms
# Tokens/cycle: 640 (SLICE=1) or 1280 (SLICE=2)
# No threshold: every cycle runs, never skips. Fastest possible reactions.
# CAPTURE_FPS = float(os.getenv("CAPTURE_FPS", "5.0"))
# FRAMES_PER_INFERENCE = int(os.getenv("FRAMES_PER_INFERENCE", "10"))
# FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", "2"))
# WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "64"))
# CHANGE_THRESHOLD = float(os.getenv("CHANGE_THRESHOLD", "0"))
# INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "0.5"))
# TTS_MAX_NEW_TOKENS = int(os.getenv("TTS_MAX_NEW_TOKENS", "96"))
# TTS_PAUSE_AFTER = float(os.getenv("TTS_PAUSE_AFTER", "0.3"))

# ===========================================================================
# 4. PROMPT PROFILES — AI personality (switchable live from web UI)
# ===========================================================================

# Default system prompt. Used by the "default" profile below.
COMMENTATOR_PROMPT = os.getenv("COMMENTATOR_PROMPT", (
    "You are a live video commentator watching a continuous stream. "
    "Rules:\n"
    "- Keep each response to 1-2 short sentences.\n"
    "- Only mention what is NEW or has CHANGED.\n"
    "- Do NOT repeat or rephrase your previous comment.\n"
    "- If nothing notable changed, respond with exactly '...' and nothing else.\n"
    "- Be specific: mention objects, actions, colors, text on screen."
))

# Predefined system prompts. Switch via the dropdown in the web UI (no restart).
PROMPT_PROFILES = {
    "default": {
        "label": "General commentator",
        "prompt": COMMENTATOR_PROMPT,
        "suggestion": "Name every object the person touches, holds, or picks up. Describe what they do with it. Ignore posture and static background.",
    },
    "sports": {
        "label": "Sports commentary",
        "prompt": (
            "You are an enthusiastic sports commentator providing play-by-play. "
            "Rules:\n"
            "- Focus on the action: who has the ball, goals, fouls, key plays.\n"
            "- Keep each response to 1-2 short, energetic sentences.\n"
            "- Only mention what is NEW or has CHANGED.\n"
            "- Do NOT repeat your previous comment.\n"
            "- If nothing notable changed, respond with exactly '...' and nothing else."
        ),
        "suggestion": "Commentate the match: goals, passes, fouls, player positions, momentum shifts.",
    },
    "security": {
        "label": "Security camera",
        "prompt": (
            "You are a security camera monitor. Report only noteworthy events. "
            "Rules:\n"
            "- Report: people entering/leaving, unusual movement, objects left behind.\n"
            "- Keep each response to 1 short factual sentence.\n"
            "- Do NOT describe static background or normal activity.\n"
            "- Do NOT repeat your previous observation.\n"
            "- If nothing noteworthy happened, respond with exactly '...' and nothing else."
        ),
        "suggestion": "Report people entering or leaving, suspicious activity, objects moved or left behind.",
    },
    "nature": {
        "label": "Nature observer",
        "prompt": (
            "You are a calm nature documentary narrator observing a live scene. "
            "Rules:\n"
            "- Describe animal behavior, weather changes, and environmental details.\n"
            "- Keep each response to 1-2 sentences in a calm, observational tone.\n"
            "- Only mention what is NEW or has CHANGED.\n"
            "- Do NOT repeat your previous observation.\n"
            "- If nothing notable changed, respond with exactly '...' and nothing else."
        ),
        "suggestion": "Describe animal behavior, bird activity, weather changes, and movement in the scene.",
    },
    "descriptive": {
        "label": "Detailed description",
        "prompt": (
            "You are providing detailed visual descriptions of a live video stream. "
            "Rules:\n"
            "- Describe what you see thoroughly: objects, people, text, colors, layout.\n"
            "- Keep each response to 2-3 sentences with rich detail.\n"
            "- Focus on what has CHANGED since your last description.\n"
            "- Do NOT repeat your previous description.\n"
            "- If nothing changed, respond with exactly '...' and nothing else."
        ),
        "suggestion": "Describe everything you see in detail: people, objects, text, colors, layout changes.",
    },
}

# ===========================================================================
# 5. SERVER & STREAMING — network, display, sync
# ===========================================================================

# Host: 127.0.0.1 = localhost only. 0.0.0.0 = allow LAN access.
# WARNING: no authentication. Don't expose on untrusted networks.
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8199"))

# JPEG quality for /api/frame snapshot endpoint (1-100).
FRAME_JPEG_QUALITY = int(os.getenv("FRAME_JPEG_QUALITY", "80"))

# Display frame rate for the /api/mjpeg browser stream.
# This is the DISPLAY rate, not the AI capture rate (that's CAPTURE_FPS above).
# 10 = smooth enough for a preview. Higher = smoother but more bandwidth.
MJPEG_FPS = int(os.getenv("MJPEG_FPS", "10"))

# Adaptive sync: delays the video stream so commentary matches what you see.
# 5.0 = video is 5 seconds behind real-time (initial, adapts after first cycle).
# 0   = no delay, real-time video (commentary will lag behind what you see).
STREAM_DELAY_INIT = float(os.getenv("STREAM_DELAY_INIT", "5.0"))

# How fast the adaptive delay adjusts. 0.2 = adapts within ~5 cycles.
# Higher = reacts faster but jittery. Lower = smoother but slow to adapt.
STREAM_DELAY_EMA_ALPHA = float(os.getenv("STREAM_DELAY_EMA_ALPHA", "0.2"))

# ===========================================================================
# 6. TTS PATHS — model files for text-to-speech (rarely change)
# ===========================================================================

# Enable TTS. Adds ~0.6-0.7 GB VRAM for the vocoder.
ENABLE_TTS = os.getenv("ENABLE_TTS", "false").lower() == "true"

# Vocoder model directory (inside the model folder).
TTS_MODEL_DIR = os.getenv("TTS_MODEL_DIR", os.path.join(MODEL_PATH, "assets", "token2wav"))

# Reference audio for voice cloning. WAV file, loaded at 16kHz mono.
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", os.path.join(MODEL_PATH, "assets", "HT_ref_audio.wav"))

# Float16 vocoder: saves VRAM but currently crashes. Keep false.
TTS_FLOAT16 = os.getenv("TTS_FLOAT16", "false").lower() == "true"
