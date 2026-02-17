# Sprint 1 Plan: MVP Video Chat with AI

## Context

The project foundation is complete (docs, agents, git repo). We need to build the first working prototype: a continuous live video stream monitor where the AI watches along and narrates what it sees, steerable by the user mid-stream. This is a PoC, not production-grade.

The repo-researcher analyzed all 10 demos across both cloned repos. None match our "continuous monitoring loop" pattern -- they're all either request-response (file upload) or voice-centric. We build our own application but reuse proven patterns from the Gradio demo (model loading, streaming inference, SSE output).

## Application Structure

```
app/
├── __init__.py
├── main.py              # FastAPI entry point + endpoints
├── config.py            # All configuration (env var overridable)
├── model_server.py      # Model loading + inference function
├── frame_capture.py     # OpenCV frame capture (background thread)
├── sliding_window.py    # Ring buffer for recent frames
├── monitor_loop.py      # Orchestrator: capture -> window -> inference -> output
├── static/
│   └── index.html       # Web UI (vanilla HTML/JS/CSS, no build step)
└── requirements.txt
scripts/
├── setup_env.sh         # Conda env creation
└── start.sh             # Launch script
```

## Implementation Steps

### Step 1: Environment Setup

Create conda env `video_chat` (Python 3.11), install dependencies.

Dependencies (pinned versions from `MiniCPM-V-CookBook/demo/web_demo/gradio/server/requirements.txt`):
- `torch==2.7.1`, `torchvision==0.22.1` (CUDA 12.x)
- `transformers==4.55.0`, `accelerate==1.9.0`
- `fastapi==0.116.1`, `uvicorn==0.35.0`
- `opencv-python-headless` (frame capture)
- `pillow==11.3.0`, `numpy==2.2.6`
- `einops`, `timm`, `safetensors`, `tokenizers`, `triton` (model dependencies)

Download model: **Full BF16** from [HuggingFace](https://huggingface.co/openbmb/MiniCPM-o-4_5) (`openbmb/MiniCPM-o-4_5`, ~18.7GB, 4 safetensor files).
Fallback if VRAM issues: [GGUF](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf) variants (Q8_0: 8.7GB, Q6_K: 6.7GB) -- requires switching to llama.cpp inference.

**Verify:** `python -c "import torch; print(torch.cuda.is_available())"` and `nvidia-smi`.

---

### Step 2: Model Server (`model_server.py`)

Load MiniCPM-o 4.5 in vision-only mode. Expose a single `infer()` function.

**Reuse from:** `MiniCPM-V-CookBook/demo/web_demo/gradio/server/models/minicpmo4_5.py`
- Vision-only config: `init_audio=False`, `init_tts=False`, `init_vision=True`
- `attn_implementation='sdpa'`, `torch_dtype=torch.bfloat16`

**Inference function:**
- Input: list of PIL Images (frames) + instruction string
- Message format: `[{'role': 'user', 'content': frames + [instruction]}]`
- Video params: `use_image_id=False`, `max_slice_nums=1` (64 tokens/frame), `max_new_tokens=512`
- Stream mode: `stream=True`, `num_beams=1`, `do_sample=True`
- Suppress thinking: `suppress_tokens=[151667]`
- Returns: generator of text chunks

**Verify:** Standalone test script -- load model, infer on a test image, confirm text output + check VRAM with `nvidia-smi`.

---

### Step 3: Frame Capture (`frame_capture.py`)

Background thread that captures frames from a video source via OpenCV.

- `cv2.VideoCapture(source)` -- works with both device ID (webcam) and file path
- Target 1 FPS capture (configurable)
- Converts BGR->RGB, outputs PIL Images
- Video files loop automatically (for testing)
- Thread-safe latest_frame access
- Callback `on_frame` for pushing to the sliding window

**Verify:** Run standalone, confirm PIL Images at ~1 FPS, test with webcam + local video file.

---

### Step 4: Sliding Window (`sliding_window.py`)

Ring buffer (deque) holding the last N frames with timestamps.

- `max_frames=16` default (16 sec at 1 FPS = 1024 tokens, safe within 8192 context)
- `get_frames(n)` returns last n frames as PIL Images
- Thread-safe (capture thread writes, inference reads)
- Old frames auto-evicted by deque maxlen

---

### Step 5: Monitor Loop (`monitor_loop.py`)

The core orchestrator -- ties everything together.

**Two modes:**
- **IDLE:** Frames captured and buffered, no inference (waiting for user instruction)
- **ACTIVE:** Periodic inference with current instruction

**Inference cycle (every ~5 seconds):**
1. Check: has instruction? not already generating? enough frames?
2. Grab last 8 frames from sliding window
3. Run `model.infer(frames, instruction, stream=True)` in a thread pool (`asyncio.to_thread`)
4. Stream text chunks to output queue
5. Wait for next cycle

**User steering:**
- `set_instruction("describe what's happening")` -- starts monitoring
- `set_instruction("only focus on the dog")` -- changes focus, next cycle uses new instruction
- New instruction triggers immediate inference if model is idle
- Instruction persists until changed

**Key params (configurable in config.py):**
- `inference_interval=5.0` seconds between cycles
- `frames_per_inference=8` frames per batch
- Both tunable after testing actual model performance

---

### Step 6: FastAPI Server (`main.py`)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve web UI (static/index.html) |
| GET | `/api/status` | Model loaded? Capture running? Current instruction? |
| POST | `/api/start` | Start capture: `{"source": 0}` or `{"source": "/path/to/video.mp4"}` |
| POST | `/api/stop` | Stop capture |
| POST | `/api/instruction` | Set/update instruction: `{"instruction": "describe what's happening"}` |
| GET | `/api/stream` | SSE endpoint -- streams model output chunks |
| GET | `/api/frame` | Latest frame as JPEG (for video display in UI) |

SSE pattern reused from `MiniCPM-V-CookBook/demo/web_demo/gradio/server/gradio_server.py` lines 134-177.

**Startup:** Load model (30-60s) -> start server -> wait for user to start capture via UI.

---

### Step 7: Web UI (`static/index.html`)

Single HTML file, vanilla JS, no framework, no build step.

```
+------------------------------------------------------------------+
|  Video Chat with AI                                    [Status]   |
+------------------------------------------------------------------+
|                           |                                       |
|     Video Preview         |         AI Commentary                 |
|     (polling /api/frame)  |         (SSE /api/stream)             |
|                           |                                       |
|     [Start Webcam]        |   [Commentary entries stream here,    |
|     [Open Video File]     |    grouped by inference cycle]        |
|                           |                                       |
+---------------------------+---------------------------------------+
|  Instruction: [________________________________________] [Send]   |
+------------------------------------------------------------------+
```

- Video: poll `/api/frame` every 500ms, update `<img>` src
- Commentary: `EventSource('/api/stream')`, append chunks to current block
- Instruction: text input + send button, POST to `/api/instruction`

---

### Step 8: Integration Test

1. Start server, open browser to `http://localhost:8000`
2. Click "Start Webcam" -- verify video appears in preview
3. Type "describe what you see" -- verify AI commentary starts appearing every ~5 seconds
4. Change instruction to "only mention if something moves" -- verify AI changes focus
5. Test with a local video file instead of webcam
6. Monitor VRAM with `nvidia-smi` throughout

---

## Streaming App Research (for later implementation)

Part of Sprint 1 scope is evaluating which app works best for providing external video streams. Research (not implement) during Sprint 1:

- **FFmpeg + v4l2loopback**: `ffmpeg -re -i video.mp4 -f v4l2 /dev/video2` -- scriptable, no GUI
- **VLC + v4l2loopback**: VLC can output to v4l2 devices via settings
- **OBS + v4l2loopback**: Virtual camera output built-in
- v4l2loopback kernel module setup requirements

Document findings in `docs/` for Sprint 2 implementation. Sprint 1 uses direct OpenCV capture (webcam + files) which covers the PoC use case.

## Key Reference Files

| What | File |
|------|------|
| Model loading pattern | `MiniCPM-V-CookBook/demo/web_demo/gradio/server/models/minicpmo4_5.py` |
| SSE streaming pattern | `MiniCPM-V-CookBook/demo/web_demo/gradio/server/gradio_server.py:134-177` |
| Multi-frame API | `MiniCPM-V-CookBook/inference/minicpm-v4_5_video_understanding.md` |
| Dependency versions | `MiniCPM-V-CookBook/demo/web_demo/gradio/server/requirements.txt` |

## Risks

| Risk | Mitigation |
|------|------------|
| OOM with multi-frame input | Start with 8 frames + `max_slice_nums=1`. Reduce if needed. |
| Model too slow for 5s cycle | Reduce `max_new_tokens` to 256, reduce frames to 4, accept longer interval. |
| `model.chat()` API differences vs docs | Test exact API in Step 2 before building the loop. |
| Model blocks async event loop | All model calls via `asyncio.to_thread()`. |

## Configuration Defaults

```
MODEL_PATH=models/MiniCPM-o-4_5
CAPTURE_FPS=1.0
WINDOW_SIZE=16
FRAMES_PER_INFERENCE=8
INFERENCE_INTERVAL=5.0
MAX_NEW_TOKENS=512
SERVER_PORT=8000
```

All overridable via environment variables. Tuned after testing real performance.
