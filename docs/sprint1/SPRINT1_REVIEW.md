# Sprint 1 Review

## Summary

Sprint 1 delivered a working MVP: a continuous live video monitor where MiniCPM-o 4.5 watches a video stream and provides real-time AI commentary, steerable mid-stream by the user. All 8 planned steps were completed and tested.

**What was built:**
- Model server loading MiniCPM-o 4.5 in vision-only mode with streaming inference
- Background frame capture (OpenCV) with thread-safe sliding window
- Async monitor loop orchestrator with IDLE/ACTIVE modes and pub/sub output
- FastAPI server with REST + SSE endpoints
- Vanilla HTML/JS web UI with split-panel layout (video preview + AI commentary)
- Commentator-style prompting with context carry-over and scene change detection
- Centralized configuration with environment variable overrides

## Architecture

```
Video Source  --[1 FPS]--> Frame Capture --[push]--> Sliding Window
                                                         |
                                                    [get 8 frames]
                                                         v
Web UI <--[SSE]-- FastAPI <--[pub/sub]-- Monitor Loop --[infer]--> Model Server
  |                  ^                                              (MiniCPM-o 4.5)
  |  [instruction]   |
  +------------------+
```

## Key Findings

### Performance

| Metric | Value |
|--------|-------|
| Model load time | 3.9-5.7s |
| VRAM usage | 16.4 GB allocated (of 24 GB on RTX 4090) |
| Inference per cycle | 1.2-2.3s (short commentary) |
| End-to-end latency | 8-11s |
| Frame capture overhead | Negligible at 1 FPS |

### Latency Breakdown

| Component | Time | Notes |
|-----------|------|-------|
| Frame age (oldest of 8 at 1 FPS) | ~7s | Biggest contributor. Reduce with fewer frames or higher FPS. |
| Inference | ~1.3-2.3s | Depends on response length. Short prompts = fast inference. |
| Cycle interval wait | ~0-5s | Time until next cycle starts. |
| **Total** | **~8-11s** | From oldest frame capture to response delivery. |

### Model Observations

- Correctly identifies scene contents, objects, colors, text on screen, and actions
- Tracks scene changes across frames (characters, settings, transitions)
- Weak on counting (often wrong about exact numbers of objects)
- Non-deterministic with `do_sample=True` -- same scene produces different wording each time
- Context carry-over ("your last comment was...") effectively prevents repetition
- Short instructions produce short answers and fast inference cycles
- Scene change detection (pixel diff on 64x64 thumbnails) reliably skips static frames

### Architecture Observations

- Pub/sub pattern works well for multi-consumer output (SSE, test scripts, future WebSocket)
- `asyncio.Event` for immediate cycle trigger on instruction change is responsive
- Thread pool executor cleanly separates blocking model inference from async event loop
- Single-file vanilla HTML/JS is sufficient for PoC but not scalable for Sprint 2 features
- Centralized `config.py` with env var overrides makes tuning easy without code changes

## Bugs Fixed

1. **Model streaming crash** -- `model.chat(stream=True)` fell through to TTS post-processing. Fixed with a one-line patch in model code. See [model_patches.md](../model_patches.md).
2. **Sequential frame reading** -- OpenCV reads video files frame-by-frame, not by time. At 25 FPS source and 1 FPS capture, 5 reads covered only 0.2s of video. Fixed by skipping frames with `grab()`.
3. **Async race condition** -- `asyncio.create_task()` doesn't execute immediately. Consumers saw `_running=False` and exited. Fixed with `_started` Event + `wait_started()` method.
4. **Port conflict** -- Default port 8000 was in use by Portainer. Changed to 8199.

## Tuning Options

All tunable via environment variables, no code changes needed:

```bash
# Lower latency: fewer frames, higher capture rate
FRAMES_PER_INFERENCE=4 CAPTURE_FPS=2 python -m app.main

# Shorter responses
MAX_NEW_TOKENS=256 python -m app.main

# More aggressive change detection (skip more static cycles)
CHANGE_THRESHOLD=10 python -m app.main

# Less aggressive (for mostly-static scenes like security cameras)
CHANGE_THRESHOLD=2 python -m app.main
```

## What Worked Well

- **Iterative testing at each step** -- catching bugs early (streaming crash, frame reading, async race) before they compounded
- **OpenCV for PoC** -- simple, no extra dependencies, covers all MVP sources
- **Commentator prompting** -- dramatic improvement in response quality and relevance
- **Centralized config** -- easy to experiment with different parameters

## Limitations

- **~8-11s latency** -- dominated by frame age (7s for 8 frames at 1 FPS). Acceptable for commentary but not for real-time conversation.
- **No real video playback in UI** -- frame polling at 500ms gives a slideshow feel, not smooth video
- **No audio** -- vision-only mode, no TTS/STT
- **No Docker** -- runs directly on host, not reproducible for others
- **No persistent configuration** -- env vars must be set each run (no .env file support)
- **Model counting weakness** -- numbers of objects are often wrong
- **No reconnection logic** -- if capture stops (camera disconnect, stream timeout), manual restart needed

## Sprint 2 Ideas

Based on findings and user feedback during Sprint 1:

### High Priority
- **Docker setup** -- containerize for reproducibility and easier deployment
- **Real video playback** -- actual `<video>` element with WebRTC or HLS, not frame polling
- **Nicer web UI** -- React or Vue frontend with proper controls and styling
- **VLC/RTSP stream input** -- v4l2loopback setup, stream URL support (partially prepared in UI)
- **Lower latency** -- experiment with fewer frames (4), higher FPS (2), and adaptive intervals

### Medium Priority
- **TTS output** -- enable model's built-in TTS for spoken commentary (needs WebSocket for audio streaming)
- **STT input** -- voice instructions instead of typing
- **WebSocket transport** -- replace SSE for bidirectional communication (needed for audio)
- **Persistent config** -- `.env` file support or config UI
- **Auto-reconnect** -- capture restarts on source failure

### Lower Priority / Research
- **Multi-model pipeline** -- feed vision output into other LLMs or alert systems
- **GGUF quantized model** -- for lower VRAM GPUs (needs llama.cpp backend)
- **Phone camera input** -- remote video from mobile device
- **Recording/logging** -- save commentary + timestamps for later review
- **Alert system** -- trigger notifications on specific events ("tell me when someone enters")

## Files Created/Modified in Sprint 1

| File | Purpose |
|------|---------|
| `app/__init__.py` | Package marker |
| `app/config.py` | All configuration, env var overridable |
| `app/model_server.py` | Model loading + streaming inference |
| `app/frame_capture.py` | Background thread capture (OpenCV) |
| `app/sliding_window.py` | Thread-safe ring buffer with FrameMeta |
| `app/monitor_loop.py` | Async orchestrator with pub/sub |
| `app/main.py` | FastAPI server with 7 endpoints |
| `app/static/index.html` | Web UI (vanilla HTML/JS/CSS) |
| `app/requirements.txt` | Python dependencies |
| `scripts/test_model.py` | Model loading + inference test |
| `scripts/test_capture.py` | Frame capture test |
| `scripts/test_monitor.py` | End-to-end monitor loop test |
| `docs/model_patches.md` | Model bug documentation and fix |
| `docs/lessons_learned.md` | Development lessons for future reference |
