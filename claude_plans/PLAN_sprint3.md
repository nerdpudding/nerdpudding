# Sprint 3 Plan: Docker, WebRTC, Input Robustness

## Context

Sprint 2 delivered a complete end-to-end pipeline: video in, text + TTS audio out, with adaptive pacing and scene-weighted commentary density. The repo is now public as **NerdPudding** on GitHub.

Sprint 3 focuses on: making it easy for others to run (Docker), enabling browser-native video input (LiveKit WebRTC), handling real-world input failures (robustness), and polishing the UI to support both paths.

**Two installation paths** are maintained throughout:
1. **Conda/pip** (current) — for development and users who prefer native installs
2. **Docker** — for quick setup, reproducibility, and deployment

Both paths use the same app code, same config, same env vars. Docker is a packaging layer, not a fork.

## Development approach

- All work on `sprint3` branch, merge to `main` via PR when complete
- Each step gets its own detailed implementation plan before coding
- Test after each step, both Docker and non-Docker paths
- External review of plans before implementation (as per Sprint 2 workflow)

## Steps

### Step 1: Docker Setup

**Goal:** Containerize the app with GPU passthrough. Model files as bind mount, not baked into image.

**Key decisions:**
- Base image: `nvidia/cuda:12.6.0-runtime-ubuntu24.04` + Python 3.12
- Two services in docker-compose: `app` (GPU, port 8199) + `livekit` (port 7880, optional profile)
- Model dir as read-only bind mount (`./models:/app/models:ro`)
- Non-root user in container (security)
- `.env.example` with all documented config vars
- `SERVER_HOST=0.0.0.0` inside Docker (container networking requires it)

**Deliverables:**
- `Dockerfile`
- `docker-compose.yml`
- `.env.example`
- `.dockerignore`
- Updated README with Docker quick start
- Updated tuning guide with Docker notes

**Verification:**
- `docker compose build` succeeds
- `docker compose up app` loads model, serves on 8199, browser works
- `docker compose down` clean shutdown
- Non-Docker path (`python -m app.main`) still works unchanged

**Detailed plan:** `claude_plans/PLAN_sprint3_dockerization.md` (created separately)

---

### Step 2: LiveKit WebRTC

**Goal:** Enable browser webcam as video input via LiveKit, TTS audio back to browser via LiveKit audio track.

**Key decisions:**
- LiveKit server runs as Docker container (port 7880)
- Python bot joins LiveKit room, subscribes to video, publishes TTS audio
- FrameProvider abstraction: OpenCVProvider (existing) + LiveKitProvider (new)
- MonitorLoop stays source-agnostic (reads from SlidingWindow regardless)
- Frontend uses `livekit-client` from CDN (no build step)
- LiveKit is opt-in: `LIVEKIT_ENABLED=false` = everything works as before

**Deliverables:**
- `app/livekit_bot.py` — bot that joins rooms, extracts frames, publishes audio
- `app/frame_provider.py` — abstract interface + OpenCV/LiveKit implementations
- `config/livekit.yaml` — LiveKit server config
- Updated `docker-compose.yml` with LiveKit service
- Token endpoint (`POST /api/livekit/token`)
- Frontend LiveKit integration

**Depends on:** Step 1 (LiveKit server in Docker)

**Detailed plan:** Created when Step 1 is complete.

---

### Step 3: Input Robustness

**Goal:** Reliable handling of all input sources with auto-reconnect and clear error reporting.

**Key decisions:**
- Reconnect logic in FrameCapture: exponential backoff (1s to 30s)
- Enhanced `/api/status` with `capture_healthy`, `livekit_connected`
- Clear error messages for common failures (source not found, permission denied, stream timeout)
- Test matrix with real sources (RTSP, phone camera, webcam)

**Deliverables:**
- Reconnect logic in `app/frame_capture.py`
- Config: `RECONNECT_DELAY`, `MAX_RECONNECT_DELAY`
- Enhanced status endpoint
- Manual test results documented

**Depends on:** Step 2 (all input paths exist)

**Detailed plan:** Created when Step 2 is complete.

---

### Step 4: UI Updates

**Goal:** Polish the web UI to support both OpenCV and LiveKit modes with proper controls and status.

**Key decisions:**
- Source mode selector: "File/RTSP/Device" (OpenCV) vs "Browser Webcam" (LiveKit)
- Status indicators for capture health, LiveKit connection, model/TTS status
- Volume slider for TTS audio
- Still vanilla HTML/JS/CSS (React/Vue deferred to Sprint 4)
- Graceful degradation: LiveKit UI hidden when LiveKit not available

**Deliverables:**
- Updated `app/static/index.html`
- Source mode selector with conditional UI
- Status indicators
- TTS volume control

**Depends on:** Steps 2-3 (both input paths working and robust)

**Detailed plan:** Created when Step 3 is complete.

---

## Dependencies

```
Step 1: Docker ──────────────────────────────┐
Step 2: LiveKit (depends on Step 1) ─────────┤
Step 3: Input Robustness (depends on Step 2) ┤
Step 4: UI Updates (depends on Steps 2-3) ───┘
```

## Dual-path contract

Every step must verify BOTH paths:
- Docker: `docker compose up` → browser works
- Native: `python -m app.main` → browser works

If a step breaks the native path, it's not done.

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Docker GPU passthrough issues | Can't run model in container | Test early with minimal container. NVIDIA Container Toolkit required — document prerequisite. |
| LiveKit adds too much complexity | Delays sprint, hard to maintain | LiveKit is opt-in. Steps 1+3+4 deliver value without it. Can defer to Sprint 4 if needed. |
| autoawq custom fork doesn't build in Docker | Model won't load | Test in Docker early (Step 1). Pin commit hash, not just branch. |
| CUDA version mismatch in container | PyTorch can't find GPU | Match base image CUDA version to host driver compatibility. Document minimum driver version. |
| LiveKit server resource usage | Extra RAM/CPU on host | LiveKit server is lightweight (~50 MB RAM). Profile during testing. |

## Config additions (all steps combined)

```python
# LiveKit (Step 2)
LIVEKIT_ENABLED = os.getenv("LIVEKIT_ENABLED", "false").lower() == "true"
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

# Reconnect (Step 3)
RECONNECT_DELAY = float(os.getenv("RECONNECT_DELAY", "1.0"))
MAX_RECONNECT_DELAY = float(os.getenv("MAX_RECONNECT_DELAY", "30.0"))
```

## New files (all steps combined)

| File | Step | Purpose |
|------|------|---------|
| `Dockerfile` | 1 | App container (CUDA + Python) |
| `docker-compose.yml` | 1, 2 | App + LiveKit orchestration |
| `.env.example` | 1 | Documented config defaults |
| `.dockerignore` | 1 | Exclude models, cache, repos from build context |
| `app/frame_provider.py` | 2 | Abstract FrameProvider + OpenCV/LiveKit implementations |
| `app/livekit_bot.py` | 2 | LiveKit room bot: video subscribe, audio publish |
| `config/livekit.yaml` | 2 | LiveKit server configuration |
