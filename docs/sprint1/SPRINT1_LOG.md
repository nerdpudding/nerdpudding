# Sprint 1 Log

Progress log for Sprint 1 MVP implementation. Documents setup steps, test results, and findings so everything is reproducible.

## Step 1: Environment Setup

### Conda environment

```bash
conda create -n video_chat python=3.12 -y
conda activate video_chat
pip install -r app/requirements.txt
```

Python 3.12 chosen because the model was built for `transformers==4.51.0` (see `config.json: "transformers_version": "4.51.0"`). Originally tried 4.55.0 from the CookBook requirements but it had breaking API changes (`DynamicCache.seen_tokens` removed, `generate()` return type changed).

### Model download

```bash
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir models/MiniCPM-o-4_5
```

- Full BF16, ~19 GB total (4 safetensor files + model code + TTS assets)
- Downloaded to `models/` inside the project (not `~/models/`), for Docker bind-mount compatibility later
- Model includes Python files (custom architecture, `trust_remote_code=True` required)
- Security scan of `.py` files: clean, no network calls during inference, no data exfiltration

### Model patch required

The downloaded model has a bug: `model.chat(stream=True)` crashes because `chat()` doesn't short-circuit for streaming and falls through to TTS post-processing that expects non-streaming output.

**Fix:** One-line patch in `models/MiniCPM-o-4_5/modeling_minicpmo.py` (after line ~1195).

See [docs/model_patches.md](../model_patches.md) for full details and how to verify/reapply.

### HuggingFace cache

`HF_HOME` is set to `models/.hf_cache/` (inside the project) to avoid polluting `~/.cache/huggingface/`. This is done in `app/model_server.py` before importing transformers.

### Verification

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
# True 12.6 NVIDIA GeForce RTX 4090
```

---

## Step 2: Model Server

### Files created

- `app/__init__.py` -- package marker
- `app/config.py` -- all configuration, environment variable overridable
- `app/model_server.py` -- model loading (vision-only) + `infer()` function
- `scripts/test_model.py` -- standalone test script

### Key design decisions

- Vision-only mode: `init_audio=False, init_tts=False, init_vision=True` (saves ~2-4 GB VRAM)
- `CUDA_VISIBLE_DEVICES` set in app code (not conda env), configurable via env var
- `HF_HOME` redirected to project-local `models/.hf_cache/`
- Streaming inference via `TextIteratorStreamer` (background thread in model code)
- `<|im_end|>` tokens filtered from output

### Test result

```bash
python -m scripts.test_model --image test_files/images/test.jpg
```

| Metric | Value |
|--------|-------|
| Model load time | 5.7s |
| Inference time (1 frame, full response) | 11.5s |
| Response length | 1759 chars |
| VRAM allocated | 16.4 GB |
| VRAM reserved | 16.7 GB |
| VRAM free | ~7.5 GB (of 24 GB) |

The model correctly described a Raspberry Pi robot car from the test image, with detailed component identification. Streaming worked -- text chunks arrived progressively.

### Performance notes

- 11.5s was for a long, unconstrained response (~512 tokens). The monitoring loop targets shorter updates with fewer `max_new_tokens`, so cycle time should be lower.
- 7.5 GB VRAM headroom is sufficient for multi-frame context (8 frames at 64 tokens/frame = 512 tokens).
- `transformers==4.51.0` shows a deprecation warning about `seen_tokens` -- harmless, will go away when/if the model code is updated upstream.

---

## Step 3: Frame Capture

### Library evaluation

Evaluated video capture libraries for our requirements (webcam, video files, later RTSP/v4l2loopback, 1 FPS, PIL output, background thread, Docker compatible):

| Library | Strengths | Weaknesses | Verdict |
|---------|-----------|------------|---------|
| **OpenCV** (`cv2.VideoCapture`) | Simplest API, already installed, webcam + files + v4l2 + RTSP | No built-in threading | **Chosen for PoC** |
| **VidGear/CamGear** | Built-in threaded capture, better stream protocol support | Extra dependency, threading optimization irrelevant at 1 FPS | Consider for Sprint 2 |
| **ffmpegcv** | Lightweight FFmpeg wrapper, GPU decode, good stream support | Less established, extra dependency | Interesting alternative for later |
| **GStreamer** | Most flexible, native HW acceleration plugins | Steep learning curve, overkill for PoC | No |

**Decision: OpenCV.** At 1 FPS capture rate, frame capture performance is irrelevant -- model inference (11.5s) is the bottleneck, not reading a single frame. OpenCV covers all MVP sources (webcam, video files) and also supports v4l2loopback devices and RTSP for Sprint 2. Threading is a simple daemon thread wrapper. The modular design (`frame_capture.py` behind a clean interface) allows swapping the backend later without affecting the rest of the app.

### Files created

- `app/frame_capture.py` -- background thread capture with `on_frame` callback
- `app/sliding_window.py` -- thread-safe ring buffer (deque, max 16 frames)
- `scripts/test_capture.py` -- standalone test for capture + window

### Bug found and fixed: sequential frame reading for video files

OpenCV's `cv2.VideoCapture.read()` reads frames sequentially for video files -- each call returns the next frame regardless of elapsed time. At 25 FPS source and 1 FPS capture, 5 reads in 5 seconds only covered the first 0.2 seconds of video.

**Fix:** For video files, skip `src_fps / CAPTURE_FPS` frames per interval using `grab()` (fast, no decode) before `read()` (decode only the target frame). This simulates real-time playback. Not needed for live sources (webcam) where `read()` always returns the latest frame.

### Test result

```bash
python -m scripts.test_capture --source test_files/videos/test.mp4
```

- Test video: 9.6s, 25.1 FPS, 1264x720
- Auto-detected video duration, captured for full length
- 10 frames captured at 1 FPS, evenly spread across the video
- All frames saved to `test_files/videos/capture_test/` for visual verification
- Frame quality matches source video (no additional compression)
- Sliding window correctly held all frames with push/get working across threads

---

## Step 4: Sliding Window

Implemented together with Step 3. Simple `collections.deque` with `maxlen=WINDOW_SIZE` (default 16). Thread-safe via `threading.Lock`. Capture thread pushes, inference loop reads. Old frames auto-evicted.

No separate test needed -- validated as part of the capture test above.

---

## Step 5: Monitor Loop

### Files created

- `app/monitor_loop.py` -- async orchestrator with IDLE/ACTIVE modes
- `scripts/test_monitor.py` -- standalone end-to-end test (model + capture + loop)

### Design

The monitor loop is the core orchestrator that ties frame capture, sliding window, and model server together. It runs as an async task and periodically grabs frames from the sliding window to run inference.

**Two modes:**
- **IDLE** -- frames captured and buffered, no inference (waiting for user instruction)
- **ACTIVE** -- periodic inference with current instruction every `INFERENCE_INTERVAL` seconds

**Key patterns:**
- `asyncio.Event` for immediate cycle trigger on instruction change
- Pub/sub pattern for streaming text chunks to multiple consumers (SSE, WebSocket, test scripts)
- `loop.run_in_executor()` for non-blocking model inference (blocking call in thread pool)
- `loop.call_soon_threadsafe()` to push chunks from inference thread to async queue
- `None` sentinel in queue marks end of an inference cycle
- `_started` event + `wait_started()` to synchronize consumers with the loop startup

### Bug found and fixed: async race condition

`asyncio.create_task(monitor.run())` schedules the task but doesn't execute it until the current coroutine yields. This caused two problems:
1. `stream()` checked `_running` (still `False`) and exited immediately -- no output
2. `stop()` called before `run()` started, then `run()` overrode `_running` back to `True` -- infinite loop

**Fix:** Added `_started` asyncio.Event that `run()` sets when it begins, and `wait_started()` for consumers to await. Added `_stop_requested` flag to prevent `run()` from starting after `stop()`.

### Test result

```bash
python -m scripts.test_monitor --source test_files/videos/test.mp4 --cycles 2
```

Test video: 9.6s animated scene of a cat protesting with police dogs (loops automatically).

| Metric | Cycle 1 | Cycle 2 | Cycle 3 (new instruction) |
|--------|---------|---------|---------------------------|
| Instruction | "describe what you see" | "describe what you see" | "list only the colors" |
| Inference time | 7.5s | 9.3s | 1.7s |
| Output | Detailed scene description (~200 words) | Different detailed description | "Orange, brown, black, white, yellow, gray" |

**Observations:**
- Model correctly identified scene contents across 8 frames: cat on crate, "MI-AUW! NOW!" sign, police dogs, urban setting, lighting
- Each cycle produces different wording for the same scene (non-deterministic with `do_sample=True`)
- Instruction change works: "list only the colors" gave a 6-word response in 1.7s vs 7-9s for detailed descriptions
- Short instructions = short answers = fast cycles. This is useful for tuning `INFERENCE_INTERVAL` later
- Model load: 3.9s (warm, model already cached from previous tests)
- Frame accumulation: ~7s to buffer 8 frames at 1 FPS before first inference
- Video file looping works correctly -- capture continued beyond the 9.6s video duration

### Pub/sub refactor

After initial testing, refactored the monitor loop from a single `asyncio.Queue` to a pub/sub pattern (`subscribe()`/`unsubscribe()`/`_publish()`). Each consumer (SSE connection, WebSocket client, test script) gets its own independent queue via `subscribe()`. This makes it trivial to add new transport types later (WebSocket for TTS/STT, etc.) without changing the monitor loop itself.

Retested after refactor -- same behavior, no regressions.

---

## Step 6: FastAPI Server

### Files created

- `app/main.py` -- FastAPI entry point with all endpoints and lifespan management

### Design

Thin HTTP layer over the existing components. The server loads the model on startup via FastAPI's lifespan context manager, then exposes the pipeline through REST endpoints and SSE.

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve web UI (static/index.html) or placeholder |
| GET | `/api/status` | Model loaded? Capture running? Mode? Instruction? Frame count? |
| POST | `/api/start` | Start capture: `{"source": "test_files/videos/test.mp4"}` |
| POST | `/api/stop` | Stop capture, clear instruction |
| POST | `/api/instruction` | Set/change instruction: `{"instruction": "describe what you see"}` |
| GET | `/api/stream` | SSE endpoint -- each connection gets its own subscriber |
| GET | `/api/frame` | Latest frame as JPEG (quality 80) |

**Key patterns:**
- FastAPI lifespan for startup (model load, monitor loop start) and shutdown (cleanup)
- SSE via `StreamingResponse` with `text/event-stream` -- no extra dependencies
- Each SSE connection subscribes to the monitor loop's pub/sub independently
- SSE keepalive comments every 15s to prevent proxy/browser timeout
- Typed SSE events: `data:` for text chunks, `event: cycle_end` for cycle boundaries
- Frame endpoint converts PIL Image to JPEG in-memory via `io.BytesIO`
- Pydantic models for request/response validation
- Server port: 8199 (configurable via `SERVER_PORT` env var)

### Test result

Tested all endpoints with curl against the running server:

```bash
python -m app.main  # starts server on port 8199
```

| Endpoint | Result |
|----------|--------|
| `GET /api/status` | JSON response with all fields correct (IDLE, no capture, 0 frames) |
| `POST /api/start` | Capture started on test video, status updated to capture_running=true |
| `GET /api/status` (after 3s) | 14 frames buffered, still IDLE (no instruction yet) |
| `GET /api/frame` | JPEG 1264x720, valid image |
| `POST /api/instruction` | Instruction set, monitor switched to ACTIVE, inference started |
| `GET /api/stream` | SSE chunks streaming, 3 full cycles received with `cycle_end` events |
| `POST /api/stop` | Capture stopped, instruction cleared, back to IDLE |

All endpoints working correctly. SSE streaming delivers text chunks in real-time with proper event formatting.

---

## Step 7: Web UI

### Files created

- `app/static/index.html` -- single-file vanilla HTML/JS/CSS, no build step

### Design

Split-panel layout: video preview on the left (polling `/api/frame`), AI commentary on the right (SSE `/api/stream`). Controls at the bottom: source input, start/stop buttons, instruction input.

**Key features:**
- Frame polling via `new Image()` at 500ms intervals (avoids broken image flicker)
- SSE with `EventSource` -- auto-reconnect built in
- Cycle metadata displayed per commentary block (frame IDs, capture timestamps, inference time, latency)
- Source input accepts file paths, device IDs, and stream URLs (RTSP, HTTP)
- Status badge reflects IDLE/ACTIVE mode

### Test result

Opened `http://localhost:8199` in browser. Verified:
- Video frames appear and update in real-time
- Start/stop/instruction controls work
- Commentary streams in with per-cycle metadata
- Instruction change mid-stream works correctly

---

## Step 8: Integration Test & Optimization

### End-to-end test

Full pipeline tested: server startup → capture start → instruction set → streaming commentary → instruction change → stop.

**Test with short video (9.6s cat scene, looping):**
- Commentary: short, relevant, non-repeating
- Inference: ~1.2-2.3s per cycle
- Latency: ~8.6-9.4s (dominated by 8 frames × 1 FPS = 7s frame age)

**Test with longer video (animated series, multiple scenes):**
- Model correctly tracked scene changes: characters, actions, setting transitions
- Commentary stayed focused on new/changed elements
- Inference: ~1.3-1.7s per cycle
- Latency: ~8.3-9.0s

### Optimizations applied

**1. Commentator-style prompting:**
Replaced open-ended "describe what you see" with a system prompt that enforces 1-2 sentence responses, change-only focus, and no repetition. Moved to `config.py` as `COMMENTATOR_PROMPT` for easy customization.

**2. Context carry-over:**
Previous response is included in the next prompt ("Your last comment was: ...") so the model avoids repeating itself. Reset on instruction change.

**3. Scene change detection:**
Before each inference cycle, compares the newest frame to the previous cycle's frame using mean pixel difference on 64x64 thumbnails. If below `CHANGE_THRESHOLD` (default 5.0), the cycle is skipped entirely. Saves GPU time on static scenes.

**4. Centralized configuration:**
All hardcoded values moved to `config.py` with documentation. Every parameter is overridable via environment variable. Single source of truth.

### Latency breakdown

| Component | Time | Notes |
|-----------|------|-------|
| Frame age (oldest of 8 at 1 FPS) | ~7s | Reduce with fewer frames or higher FPS |
| Inference | ~1.3-2.3s | Depends on response length |
| Cycle interval wait | ~0-5s | Time until next cycle starts |
| **Total end-to-end** | **~8-11s** | From oldest frame capture to response complete |

### Tuning options (all via env vars, no code changes)

```bash
# Lower latency: fewer frames, higher capture rate
FRAMES_PER_INFERENCE=4 CAPTURE_FPS=2 python -m app.main

# Shorter responses
MAX_NEW_TOKENS=256 python -m app.main

# More aggressive change detection (skip more cycles)
CHANGE_THRESHOLD=10 python -m app.main
```

---
