import os

# GPU
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# Model
MODEL_PATH = os.getenv("MODEL_PATH", "models/MiniCPM-o-4_5")

# Inference
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
MAX_SLICE_NUMS = int(os.getenv("MAX_SLICE_NUMS", "1"))
MAX_INP_LENGTH = int(os.getenv("MAX_INP_LENGTH", "4352"))

# Frame capture
CAPTURE_FPS = float(os.getenv("CAPTURE_FPS", "1.0"))

# Sliding window
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "16"))
FRAMES_PER_INFERENCE = int(os.getenv("FRAMES_PER_INFERENCE", "8"))

# Monitor loop
INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "5.0"))

# Server
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
