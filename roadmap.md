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

## Sprint 2 — Stabilize and Improve

- [ ] Dockerize the setup for reproducibility
- [ ] Real video playback in UI (WebRTC or HLS, not frame polling)
- [ ] Nicer web frontend (React or Vue, proper controls and styling)
- [ ] VLC/RTSP stream input (v4l2loopback setup)
- [ ] Lower latency experiments (fewer frames, higher FPS, adaptive intervals)
- [ ] Persistent configuration (.env file support or config UI)
- [ ] Auto-reconnect on capture failure
- [ ] Evaluate TTS output (model's built-in text-to-speech)
- [ ] Evaluate STT input (voice instructions)

## Sprint 3 — Extended Capabilities

- [ ] Voice input and speech output integration
- [ ] WebSocket transport for bidirectional audio streaming
- [ ] Phone camera / remote video source support
- [ ] Explore multi-model pipeline (vision output feeding other systems)
- [ ] Monitoring and alerting use case prototype
- [ ] Recording / logging (save commentary + timestamps)

## Status

| Sprint | Status |
|--------|--------|
| Sprint 1 | Complete |
| Sprint 2 | Planned |
| Sprint 3 | Planned |

## Sprint 1 Results

See [Sprint 1 Review](docs/sprint1/SPRINT1_REVIEW.md) for detailed findings, performance data, and Sprint 2 recommendations.
