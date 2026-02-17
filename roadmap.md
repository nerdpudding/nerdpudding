# Roadmap

## Sprint 1 — MVP: Basic Video Chat

- [x] Research cloned repos and select best demo to build upon
- [x] Set up development environment (Python 3.12, conda, CUDA 12.6)
- [x] Download and configure MiniCPM-o 4.5 model (Full BF16, 19 GB)
- [x] Get model server running locally on RTX 4090 (16.4 GB VRAM, streaming works)
- [ ] Get web frontend running (text chat + video display)
- [ ] Test with webcam input — verify end-to-end: video in, text question, text response
- [ ] Test with local video file (v4l2loopback or direct feed)
- [ ] Document findings, limitations, and performance observations

## Sprint 2 — Stabilize and Improve

- [ ] Address limitations found in Sprint 1
- [ ] Dockerize the setup for reproducibility
- [ ] Improve video input flexibility (VLC stream, external sources)
- [ ] Evaluate voice input (if supported by chosen demo)
- [ ] Evaluate speech output / TTS

## Sprint 3 — Extended Capabilities

- [ ] Voice input and speech output integration
- [ ] Phone camera / remote video source support
- [ ] Explore multi-model pipeline (vision output feeding other systems)
- [ ] Monitoring and alerting use case prototype

## Status

| Sprint | Status |
|--------|--------|
| Sprint 1 | In progress (Step 2 of 8 complete) |
| Sprint 2 | Planned |
| Sprint 3 | Planned |
