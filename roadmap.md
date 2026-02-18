# Roadmap

## Sprint 1 — MVP: Basic Video Chat

- [x] Research cloned repos and select best demo to build upon
- [x] Set up development environment (Python 3.12, conda, CUDA 12.6)
- [x] Download and configure MiniCPM-o 4.5 model (Full BF16, 19 GB)
- [x] Get model server running locally (16.4 GB VRAM, streaming works)
- [x] Build frame capture layer (OpenCV, background thread, 1 FPS)
- [x] Build sliding window (ring buffer, 16 frames, thread-safe)
- [x] Build monitor loop orchestrator (IDLE/ACTIVE modes, async, pub/sub output)
- [x] Build FastAPI server (REST + SSE endpoints, port 8199)
- [x] Build web UI (vanilla HTML/JS, split-panel, frame polling + SSE)
- [x] Integration test with video files (commentary, latency measurement, scene change detection)
- [x] Optimize: commentator prompting, context carry-over, change detection
- [x] Centralize configuration (config.py, env var overridable)
- [x] Document findings, limitations, and performance observations

## Sprint 2 — AWQ, TTS, Audio Pacing

### Milestone: PoC Complete (after Step 2)

Steps 1-2 deliver a successful proof of concept: smooth native-rate video with real-time AI commentary, adaptively synced, on consumer GPU hardware with ~8.6 GB VRAM.

### Milestone: Sprint 2 Complete (after Step 4b)

Full end-to-end pipeline: video in, text + TTS audio out, with adaptive pacing and scene-weighted commentary density.

- [x] Step 1: AWQ INT4 model support (~8.6 GB VRAM, BF16 fallback via env var)
- [x] Step 1b: Latency optimization (frame striding, tuned defaults, ~52% latency reduction)
- [x] Step 2: MJPEG streaming with adaptive sync (native FPS display, EMA delay tracking)
- [x] Step 3: TTS integration (model's built-in streaming TTS, simplex API, 24kHz audio)
- [x] Step 4: Audio delivery pipeline (AudioManager, resampling, /api/audio-stream, Web Audio API)
- [x] Step 4b: Audio-commentary pacing (audio gate, breathing pause, token cap, scene-weighted density, "..." suppression)

## Sprint 3 — Docker, WebRTC, Input Robustness

Moved from Sprint 2 (Steps 5-8). Focuses on deployment, browser-native input, and polish.

- [x] HTTP MJPEG stream support (multipart/x-mixed-replace parser in frame_capture.py)
- [x] Prompt profiles (switchable AI personalities from web UI dropdown)
- [x] Config preset system (Beast/Sniper/Owl/Sentry combos, tuning guide)
- [x] Performance investigation (SageAttention v1/v2, Flash Attention 2 — SDPA flash is optimal)
- [x] torch.compile() optimization (modest improvement on skip responses)
- [ ] Docker setup (GPU passthrough, model as bind mount, docker-compose)
- [ ] LiveKit WebRTC (browser webcam input, TTS audio to browser)
- [ ] Input robustness (RTSP/IP cam/phone/VLC testing, auto-reconnect)
- [ ] UI updates (source mode selector, LiveKit player, TTS controls, status indicators)

## Sprint 4 — Extended Capabilities

- [ ] React or Vue frontend (proper controls, styling, scalable UI)
- [ ] STT input (voice instructions via model's built-in speech recognition)
- [ ] Multi-GPU / GGUF via llama.cpp (offload to both GPUs)
- [ ] Phone camera / remote video source support
- [ ] Multi-model pipeline (vision output feeding other LLMs or alert systems)
- [ ] Monitoring and alerting use case prototype
- [ ] Recording / logging (save commentary + timestamps)
- [ ] Persistent configuration (.env file support or config UI)
- [ ] MiniCPM-V 4.5 vision-only mode (no TTS, 3D-Resampler: 6 frames → 64 tokens, much faster)
- [ ] SGLang inference backend (RadixAttention prefix caching, ~10-20% faster)
- [ ] TensorRT-LLM inference backend (compiled model, potentially ~30-50% faster)

## Status

| Sprint | Status | Tag | Branch |
|--------|--------|-----|--------|
| Sprint 1 | Complete | `poc-milestone` | `sprint1` |
| Sprint 2 | Complete | `sprint2-milestone` | `sprint2` |
| Sprint 3 | In Progress | | `sprint3` |
| Sprint 4 | Planned | | |

## Sprint Results

- [Sprint 1 Review](docs/sprint1/SPRINT1_REVIEW.md) -- findings, performance data, Sprint 2 recommendations
- [Sprint 2 Review](docs/sprint2/SPRINT2_REVIEW.md) -- findings, performance data, Sprint 3 recommendations
- [Sprint 2 Plan](archive/2026-02-19_PLAN_sprint2.md) -- original 8-step plan (Steps 5-8 deferred to Sprint 3)
- [Sprint 2 Log](docs/sprint2/SPRINT2_LOG.md) -- progress, bugs found, performance measurements
