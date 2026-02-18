# Sprint 3 Step 1: Dockerization

## Context

NerdPudding is public on GitHub. The biggest barrier for new users is the manual setup: conda, pip, model download, patches, then run. Docker reduces this to `docker compose up` (after downloading the model).

This plan covers the app container only. LiveKit is added in Step 2 as a separate service in the same compose file.

**Both install paths must work:** Docker AND native (conda/pip). Docker wraps the same app — no code forks.

## Research: Reference repo patterns

From the CookBook's WebRTC_Demo:
- Base image: uses pre-built Docker images (not CUDA-based — their inference runs outside Docker via llama.cpp)
- LiveKit server: `livekit/livekit-server` image, port 7880, config via mounted `livekit.yaml`
- Docker compose: 3 services (frontend nginx, backend Python, livekit), no GPU passthrough (inference is native)
- Model files: mounted from host, not baked into image

**Our approach differs:** We run inference INSIDE the container (PyTorch + CUDA), so we need the NVIDIA CUDA runtime base image and GPU passthrough.

## Prerequisites (user must have)

- Docker Engine with Compose v2 (`docker compose` command)
- NVIDIA Container Toolkit (for GPU passthrough)
- NVIDIA driver compatible with CUDA 12.6
- Model files downloaded to `./models/` (same as native path)

## Files to create

### 1. `Dockerfile`

```dockerfile
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Use python3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
COPY app/requirements.txt /app/app/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /app/app/requirements.txt

# Copy application code
COPY app/ /app/app/
COPY scripts/ /app/scripts/

# Switch to non-root user
USER appuser

# Default: bind to 0.0.0.0 inside container (required for Docker networking)
ENV SERVER_HOST=0.0.0.0

EXPOSE 8199

CMD ["python", "-m", "app.main"]
```

**Key decisions:**
- `nvidia/cuda:12.6.3-runtime-ubuntu24.04` — matches host CUDA, runtime-only (smaller than devel)
- Python 3.12 from Ubuntu repos (not building from source)
- `git` needed for `autoawq` pip install from git URL
- `libgl1` + `libglib2.0-0` needed for OpenCV headless
- Non-root `appuser` (security)
- `SERVER_HOST=0.0.0.0` as ENV default (override native `127.0.0.1` for container networking)
- Model files NOT copied — mounted at runtime

### 2. `docker-compose.yml`

```yaml
services:
  app:
    build: .
    container_name: nerdpudding
    restart: unless-stopped
    ports:
      - "${SERVER_PORT:-8199}:8199"
    volumes:
      - ./models:/app/models:ro
      - ./test_files:/app/test_files:ro
    environment:
      - ENABLE_TTS=${ENABLE_TTS:-false}
      - MODEL_PATH=${MODEL_PATH:-models/MiniCPM-o-4_5-awq}
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    env_file:
      - .env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Key decisions:**
- GPU passthrough via `deploy.resources.reservations.devices` (compose v2 syntax)
- `.env` file for all config overrides (auto-loaded by compose)
- Models as read-only bind mount
- No LiveKit service yet (added in Step 2)
- `restart: unless-stopped` for convenience
- Port configurable via env var

### 3. `.env.example`

```bash
# NerdPudding Configuration
# Copy to .env and modify as needed: cp .env.example .env

# --- Model ---
# MODEL_PATH=models/MiniCPM-o-4_5-awq    # AWQ INT4 (default, ~8.6 GB VRAM)
# MODEL_PATH=models/MiniCPM-o-4_5        # BF16 (~18.5 GB VRAM)

# --- GPU ---
# CUDA_VISIBLE_DEVICES=0                  # Which GPU (0 = first)

# --- TTS ---
# ENABLE_TTS=false                        # Enable text-to-speech audio
# TTS_PAUSE_AFTER=1.0                     # Silence between audio segments (seconds)
# TTS_MAX_NEW_TOKENS=150                  # Max response length with TTS

# --- Server ---
# SERVER_HOST=127.0.0.1                   # Bind address (0.0.0.0 for Docker/network)
# SERVER_PORT=8199                        # Server port

# --- Inference tuning ---
# FRAMES_PER_INFERENCE=4                  # Frames per cycle (1-8)
# FRAME_STRIDE=2                          # Skip every Nth frame (1-4)
# CAPTURE_FPS=2.0                         # Inference capture rate
# CHANGE_THRESHOLD=5.0                    # Scene change sensitivity (0-50)
# MAX_NEW_TOKENS=512                      # Max response length (text-only)
# INFERENCE_INTERVAL=1.0                  # Min pause between cycles

# --- Video sync ---
# STREAM_DELAY_INIT=5.0                   # Initial sync delay (0 = no sync)
```

### 4. `.dockerignore`

```
MiniCPM-o/
MiniCPM-V-CookBook/
models/
.hf_cache/
__pycache__/
*.pyc
.git/
.env
.env.local
archive/
claude_plans/
concepts/
docs/
*.md
!app/requirements.txt
test_files/
.claude/
```

**Purpose:** Keep the build context small and fast. Models, reference repos, docs, and caches are excluded. Only `app/` and `scripts/` are copied.

## Changes to existing files

### `README.md` — Docker Quick Start section

Add a Docker section right after the existing Quick Start:

```markdown
### Quick Start (Docker)

```bash
# 1. Download model (same as native setup)
huggingface-cli download openbmb/MiniCPM-o-4_5-awq --local-dir models/MiniCPM-o-4_5-awq

# 2. Download TTS assets (if using TTS)
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir models/MiniCPM-o-4_5 --include "assets/*"
cp -r models/MiniCPM-o-4_5/assets models/MiniCPM-o-4_5-awq/assets

# 3. Apply model patches (see docs/model_patches.md)

# 4. Copy and edit config
cp .env.example .env
# Edit .env to enable TTS, change model, etc.

# 5. Build and run
docker compose up --build

# Server starts on http://localhost:8199
```
```

### `docs/tuning_guide.md` — Docker notes

Add a small section after the GPU-specific sections:

```markdown
### Docker

All settings work the same in Docker via the `.env` file or `docker compose` environment variables:

```bash
# Edit .env
ENABLE_TTS=true
TTS_PAUSE_AFTER=1.5

# Or override inline
ENABLE_TTS=true docker compose up
```

Note: `SERVER_HOST` is automatically set to `0.0.0.0` inside the container. You don't need to change it.
```

## Prerequisites documentation

Add to README under Prerequisites:

```markdown
**For Docker:**
- Docker Engine 24+ with Compose v2
- NVIDIA Container Toolkit ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- NVIDIA driver 535+ (for CUDA 12.6 compatibility)
```

## Verification

1. **Docker build:**
   ```bash
   docker compose build
   ```
   Must complete without errors. Image should be ~4-5 GB.

2. **Docker run (text-only):**
   ```bash
   docker compose up
   ```
   - Model loads from bind mount
   - Server accessible at http://localhost:8199
   - Browser shows UI, can start video source and get commentary

3. **Docker run (TTS):**
   ```bash
   ENABLE_TTS=true docker compose up
   ```
   - TTS audio plays in browser

4. **Docker stop:**
   ```bash
   docker compose down
   ```
   Clean shutdown, no orphan processes.

5. **Native path unaffected:**
   ```bash
   python -m app.main
   ```
   Still works exactly as before (binds to 127.0.0.1).

6. **GPU verification inside container:**
   ```bash
   docker compose run --rm app python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

## Risks

| Risk | Mitigation |
|------|------------|
| `autoawq` git install fails in Docker build | Pin to specific commit hash in requirements.txt. Test build early. |
| CUDA version mismatch (host driver vs container runtime) | Document minimum driver version (535+). Test with `nvidia-smi` inside container. |
| Large build context slows builds | `.dockerignore` excludes models, repos, cache. Context should be <10 MB. |
| OpenCV headless needs system libs | `libgl1` + `libglib2.0-0` in Dockerfile. Already tested pattern. |
| HF cache created inside container (ephemeral) | `HF_HOME` set relative to MODEL_PATH which is a bind mount. Cache persists on host. |
| `minicpmo-utils[all]` overrides torch in Docker | Same fix as native: pip install order matters, torch pinned first. May need explicit reinstall step in Dockerfile. |

## Implementation order

1. Create `.dockerignore`
2. Create `Dockerfile`
3. Create `docker-compose.yml`
4. Create `.env.example`
5. Test: `docker compose build`
6. Test: `docker compose up` (text-only)
7. Test: `ENABLE_TTS=true docker compose up`
8. Test: native path still works
9. Update README with Docker Quick Start
10. Update tuning guide with Docker notes
