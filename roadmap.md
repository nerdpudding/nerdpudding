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

## Sprint 2 — Input Agnosticism, TTS, WebRTC, Docker

### Milestone: PoC Complete (after Step 2)

Steps 1-2 deliver a successful proof of concept: smooth native-rate video with real-time AI commentary, adaptively synced, on consumer GPU hardware with ~8.6 GB VRAM. Everything from Step 3 onward is enhancement, not core validation.

- [x] AWQ INT4 model support (~8.6 GB VRAM, BF16 fallback via env var)
- [x] MJPEG streaming with adaptive sync (native FPS display, EMA delay tracking)
- [ ] TTS integration (model's built-in streaming TTS, 24kHz audio output)
- [ ] Audio delivery pipeline (AudioManager, resampling, /api/audio-stream)
- [ ] Docker setup (GPU passthrough, model as bind mount, docker-compose)
- [ ] LiveKit WebRTC (browser webcam input, TTS audio to browser)
- [ ] Input robustness (RTSP/IP cam/phone/VLC testing, auto-reconnect)
- [ ] UI updates (source mode selector, MJPEG video, TTS controls, status indicators)

## Sprint 3 — Extended Capabilities

- [ ] React or Vue frontend (proper controls, styling, scalable UI)
- [ ] STT input (voice instructions via model's built-in speech recognition)
- [ ] Multi-GPU / GGUF via llama.cpp (offload to both GPUs)
- [ ] Phone camera / remote video source support
- [ ] Multi-model pipeline (vision output feeding other LLMs or alert systems)
- [ ] Monitoring and alerting use case prototype
- [ ] Recording / logging (save commentary + timestamps)
- [ ] Persistent configuration (.env file support or config UI)

## Status

| Sprint | Status |
|--------|--------|
| Sprint 1 | Complete |
| Sprint 2 | In progress |
| Sprint 3 | Planned |

## Sprint Results

- [Sprint 1 Review](docs/sprint1/SPRINT1_REVIEW.md) -- findings, performance data, Sprint 2 recommendations
- [Sprint 2 Plan](claude_plans/PLAN_sprint2.md) -- detailed 8-step implementation plan
- [Sprint 2 Log](docs/sprint2/SPRINT2_LOG.md) -- progress, bugs found, performance measurements
