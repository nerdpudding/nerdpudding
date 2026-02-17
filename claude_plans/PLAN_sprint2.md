# Sprint 2 Plan: Input Agnosticism, TTS, WebRTC, Docker

## Context

Sprint 1 delivered a working MVP: continuous video monitoring with real-time AI commentary (text only), steerable mid-stream. It uses BF16 MiniCPM-o 4.5 (~16.4 GB VRAM on RTX 4090), OpenCV frame capture, SSE text streaming, and a vanilla HTML/JS UI with frame polling.

Sprint 2 focuses on: making the system usable with any video source (RTSP, IP cam, phone, VLC from another PC), improving UX (smooth video, TTS audio output), reducing VRAM/latency (AWQ INT4), and reproducibility (Docker). React/Vue frontend, multi-model pipelines, and multi-GPU llama.cpp are deferred to Sprint 3.

## Architecture (Sprint 2 Target)

```
Video Sources                      Server                           Browser
─────────────                      ──────                           ───────
RTSP/IP cam/VLC ──→ OpenCV ──┐
                              ├──→ SlidingWindow ──→ MonitorLoop    SSE (text)
Browser webcam ──→ LiveKit ──┘          │                │          LiveKit (audio)
                     ↕                  │          AWQ Model+TTS    MJPEG (video)
              LiveKit Server            │                │
              (Docker:7880)             │          AudioManager ──→ /api/audio-stream
                                        └──→ MJPEG endpoint ────→ <img> smooth video
```

Two input paths (OpenCV for external streams, LiveKit for browser cam) feed the same SlidingWindow. The MonitorLoop is source-agnostic. Audio output goes to LiveKit (browser sessions) and an API endpoint (programmatic clients like the robot car).

---

## Step 1: AWQ Model Support

**Goal**: Switch to INT4 AWQ model for lower VRAM and faster inference.

**Why first**: Frees ~5 GB VRAM (19→11 GB) needed for TTS (~2 GB). Also 38% faster inference.

**Key info**: `openbmb/MiniCPM-o-4_5-awq` is an official OpenBMB pre-quantized model. Same `AutoModel.from_pretrained()` loading. Needs `autoawq` from custom fork (`pip install git+https://github.com/tc-mb/AutoAWQ.git`). Uses `torch.float16` instead of `torch.bfloat16`.

**Files**:
- `app/config.py` -- No new config needed; `MODEL_PATH` already switches models
- `app/model_server.py` -- Auto-detect AWQ (check for `quantization_config` in model config), set dtype accordingly. Remove hardcoded `torch.bfloat16`.
- `app/requirements.txt` -- Add autoawq dependency

**Key change in `model_server.py`**:
```python
is_awq = hasattr(config, 'quantization_config')
dtype = torch.float16 if is_awq else torch.bfloat16
```

**Verification**:
- Download: `huggingface-cli download openbmb/MiniCPM-o-4_5-awq --local-dir models/MiniCPM-o-4_5-awq`
- Run: `MODEL_PATH=models/MiniCPM-o-4_5-awq python -m scripts.test_model --image test_files/images/test.jpg`
- Check VRAM (expect ~11 GB), compare output quality to BF16
- Verify BF16 still works: `MODEL_PATH=models/MiniCPM-o-4_5 python -m scripts.test_model --image test_files/images/test.jpg`

**Actual results (Step 1 complete)**:
- VRAM (nvidia-smi): ~8.6 GB (AWQ) vs ~18.5 GB (BF16) — 54% reduction
- Inference speed: comparable to BF16, NOT faster (vision encoder dominates, runs unquantized)
- Quality: no noticeable degradation
- Required config.json patch (modules_to_not_convert bug) and streaming patch — see docs/model_patches.md

---

## Step 1b: Latency Optimization (Frame Striding + Tuning)

**Goal**: Reduce end-to-end latency from ~10s to ~4s without losing temporal context.

**Why now**: AWQ didn't improve inference speed. Testing revealed the bottleneck is the frame capture window (8 frames × 1 FPS = 8 seconds), not inference (~2s). The INFERENCE_INTERVAL (5s idle wait) adds further delay.

**Analysis of latency**:
```
Current cycle (FRAMES_PER_INFERENCE=8, CAPTURE_FPS=1, INFERENCE_INTERVAL=5):
  Frame window:     8.0s  (8 consecutive frames, 1 apart)
  Inference:        2.0s  (AWQ or BF16, similar)
  Idle wait:        5.0s  (INFERENCE_INTERVAL, pure waste between cycles)
  End-to-end:      ~10s
```

**Three improvements**:

1. **Frame striding** (new feature): Instead of N consecutive frames, take every Kth frame from the buffer. With `FRAME_STRIDE=2`, 4 frames span 8 seconds instead of 4 seconds — same temporal coverage, half the image tokens.

2. **Lower INFERENCE_INTERVAL**: From 5.0s to 1.0s. Inference already takes ~2s, no need for extra 5s idle.

3. **Updated defaults**: `FRAMES_PER_INFERENCE=4`, `CAPTURE_FPS=2`, `FRAME_STRIDE=2`, `INFERENCE_INTERVAL=1.0`

**Files**:
- `app/config.py` -- Add `FRAME_STRIDE`, update defaults
- `app/sliding_window.py` -- Add stride parameter to `get_frames()` and `get_frames_with_meta()`
- `app/monitor_loop.py` -- Pass stride when reading frames

**Expected result**:
```
After (FRAMES_PER_INFERENCE=4, CAPTURE_FPS=2, FRAME_STRIDE=2, INFERENCE_INTERVAL=1):
  Frame window:     4.0s  (4 frames, every 2nd, spanning 4s at 2 FPS)
  Inference:        ~1.5s (fewer image tokens)
  Idle wait:        1.0s
  End-to-end:      ~4-5s
```

**Verification**:
- Run with defaults, compare cycle latency to Sprint 1 baseline
- Verify commentary quality with fewer frames (should still be coherent)
- Test with various stride values (1, 2, 3) for quality vs speed tradeoff

---

## Step 2: MJPEG Streaming (Quick UI Win)

**Goal**: Replace 500ms frame polling with smooth MJPEG stream.

**Why second**: Small change, big visible improvement. No dependencies on other steps.

**Files**:
- `app/main.py` -- Add `GET /api/mjpeg` endpoint using `multipart/x-mixed-replace`
- `app/static/index.html` -- Replace `setInterval` polling with `<img src="/api/mjpeg">`
- Keep `/api/frame` for API consumers (robot car)

**Key pattern**: `StreamingResponse` with `multipart/x-mixed-replace; boundary=frame`. Server yields JPEG frames at ~10 FPS visual update rate. Each frame boundary is `--frame\r\nContent-Type: image/jpeg\r\n\r\n<jpeg bytes>\r\n`.

**Verification**:
- Start server with video source
- Browser shows smooth video instead of slideshow
- SSE commentary still works alongside
- `/api/frame` single-shot still works

---

## Step 3: TTS Integration (Major Refactor)

**Goal**: Enable the model's built-in TTS for streaming audio output alongside text.

**Why third**: Biggest technical risk. Needs AWQ first (VRAM headroom). All subsequent steps depend on audio being available.

**Key research findings**:
- Two-phase init: (1) `init_tts=True` in config loads TTS neural weights (~700-800 MB), (2) `model.init_tts(enable_float16=True)` loads Token2wav vocoder (~600-1200 MB)
- `model.init_token2wav_cache(ref_audio_16k)` primes streaming cache (one-time per session, needs 16kHz reference audio)
- **Simplex streaming API** (NOT the duplex OmniInferenceService -- that's for voice conversation, overkill for us):
  ```python
  model.streaming_prefill(session_id=sid, msgs=[{role:"user", content:[img1, img2, text]}], is_last_chunk=True)
  for wav_chunk, text_chunk in model.streaming_generate(session_id=sid, generate_audio=True, use_tts_template=True):
      # wav_chunk = torch.Tensor float32 at 24kHz (~1 sec per chunk)
      # text_chunk = string
  ```
- When `generate_audio=False`, yields `(text_chunk, is_finished)` pairs instead
- Reference audio files available at `models/MiniCPM-o-4_5/assets/system_ref_audio.wav`
- Extra dep: `pip install minicpmo-utils[all]` (provides `stepaudio2` vocoder)
- Session resets on first prefill if `is_first=True`

**Files**:
- `app/config.py` -- Add `ENABLE_TTS`, `REF_AUDIO_PATH`, `TTS_FLOAT16`
- `app/model_server.py` -- Major refactor: add `infer_with_audio()` method using streaming_prefill/streaming_generate. Keep `infer()` for text-only backward compat.
- `app/requirements.txt` -- Add `minicpmo-utils[all]`, `soundfile`, `scipy`
- New: `scripts/test_tts.py` -- Test script: load with TTS, run one cycle, save audio to WAV

**Design**: `ModelServer` gets a new `infer_with_audio()` generator that yields `InferenceResult(text, audio_waveform, is_last)`. When `ENABLE_TTS=false`, it falls back to wrapping `infer()` output in `InferenceResult` (text only). No separate classes -- one ModelServer, two methods.

**Session handling**: Each inference cycle resets the session (new `streaming_prefill`). Context carry-over stays the same as Sprint 1 (last response included in prompt text).

**Verification**:
- `ENABLE_TTS=true MODEL_PATH=models/MiniCPM-o-4_5-awq python -m scripts.test_tts --source test_files/videos/test.mp4`
- Check VRAM (expect ~13 GB: 11 AWQ + 2 TTS)
- Listen to saved audio file for quality
- Verify text-only still works: `ENABLE_TTS=false python -m scripts.test_model --image test_files/images/test.jpg`

---

## Step 4: Audio Delivery Pipeline

**Goal**: Make TTS audio available to consumers (LiveKit, API, browser).

**Depends on**: Step 3 (TTS produces audio)

**Files**:
- New: `app/audio_manager.py` -- Receives 24kHz float32 chunks, resamples to 48kHz int16 (for WebRTC), pub/sub delivery
- `app/monitor_loop.py` -- Extend `_inference_worker` to call `infer_with_audio()` when TTS enabled, publish audio via AudioManager
- `app/main.py` -- Add `GET /api/audio-stream` endpoint (raw PCM stream for API consumers), add AudioManager to app state

**AudioManager responsibilities**:
- Accept 24kHz float32 audio from TTS
- Resample to 48kHz int16 using `scipy.signal.resample_poly` (WebRTC standard)
- Pub/sub pattern (same as MonitorLoop text) for multiple consumers
- One consumer = LiveKit audio publisher, another = API endpoint

**API endpoint**: `/api/audio-stream` returns raw PCM (48kHz, mono, int16) with format headers. The robot car use case: curl the endpoint, pipe to audio player or process programmatically.

**Verification**:
- Run with TTS enabled
- `curl localhost:8199/api/audio-stream > audio.pcm` -- save raw PCM
- `ffplay -f s16le -ar 48000 -ac 1 audio.pcm` -- listen
- Verify SSE text still works simultaneously
- Audio arrives at ~1 second intervals

---

## Step 5: Docker Setup

**Goal**: Containerize app with GPU passthrough. Prepare for LiveKit server.

**Can run in parallel with Steps 3-4**.

**Files**:
- New: `Dockerfile` -- Based on `nvidia/cuda:12.6.0-runtime-ubuntu24.04`, Python 3.12, pip install requirements + autoawq
- New: `docker-compose.yml` -- Two services: `app` (GPU, port 8199) + `livekit` (port 7880). Model dir as bind mount.
- New: `.env.example` -- Documented defaults for all config vars
- New: `config/livekit.yaml` -- LiveKit server config (adapted from CookBook reference)

**Docker design**:
- App image: ~3 GB (CUDA runtime + Python deps). Model files NOT in image.
- `docker compose up app` works standalone (no LiveKit)
- `docker compose up` starts both app + LiveKit
- `docker compose --profile livekit up` as alternative if we want LiveKit optional

**Volume mounts**: `./models:/app/models:ro`, `./test_files:/app/test_files:ro`, `./config:/app/config:ro`

**Verification**:
- `docker compose build` succeeds
- `docker compose up -d livekit` -- LiveKit runs on port 7880
- `docker compose up app` -- App loads model from volume mount, serves on 8199
- Browser at `http://localhost:8199` works
- `docker compose down` -- clean shutdown

---

## Step 6: LiveKit WebRTC

**Goal**: Browser webcam via LiveKit, TTS audio back to browser via LiveKit audio track.

**Depends on**: Steps 4 (AudioManager) and 5 (LiveKit server in Docker)

**Key research findings**:
- LiveKit Server: Go binary, Docker image `livekit/livekit-server:v1.5.3`
- Python SDK: `livekit` package, bot joins rooms as participant
- JS SDK: `livekit-client` from CDN (no npm build step, keeps vanilla HTML)
- Video frames: `rtc.VideoStream(track, format=RGB24)` → numpy → PIL Image → SlidingWindow
- Audio out: `rtc.AudioSource(48000, 1)` → `capture_frame(AudioFrame)` in 20ms chunks
- Coexists with existing FastAPI endpoints (LiveKit uses its own port 7880)

**Files**:
- New: `app/livekit_bot.py` -- Bot that joins rooms, subscribes to video, publishes TTS audio
- New: `app/frame_provider.py` -- Abstract interface. `OpenCVProvider` wraps existing FrameCapture. `LiveKitProvider` wraps the bot's video extraction. Both push to SlidingWindow.
- `app/config.py` -- Add `LIVEKIT_ENABLED`, `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `app/main.py` -- Add `POST /api/livekit/token` endpoint, bot lifecycle in lifespan
- `app/requirements.txt` -- Add `livekit`, `livekit-api`

**Frame provider abstraction** (SOLID -- dependency inversion):
```python
class FrameProvider(ABC):
    def start(self, source) -> None: ...
    def stop(self) -> None: ...
    @property
    def is_running(self) -> bool: ...
    @property
    def latest_frame(self) -> Optional[Image.Image]: ...
```
OpenCVProvider wraps existing FrameCapture. LiveKitProvider subscribes to video track. MonitorLoop reads from SlidingWindow regardless of provider -- fully source-agnostic.

**LiveKit bot flow**:
1. User clicks "Browser Webcam" in UI → frontend calls `POST /api/livekit/token`
2. Server generates JWT, creates room "monitor", bot joins as participant
3. Frontend connects to LiveKit with token, publishes webcam video track
4. Bot subscribes to video track, extracts frames at ~1 FPS, pushes to SlidingWindow
5. MonitorLoop runs inference as usual, TTS audio goes to AudioManager
6. Bot's audio publisher loop consumes from AudioManager, sends 20ms frames via `rtc.AudioSource`
7. Frontend receives audio track, plays automatically

**Frontend LiveKit** (vanilla JS, `livekit-client` from CDN):
```html
<script src="https://cdn.jsdelivr.net/npm/livekit-client@2/dist/livekit-client.umd.js"></script>
```

**Verification**:
- `LIVEKIT_ENABLED=true docker compose up`
- Browser: select "Browser Webcam" mode
- Grant camera permission → frames appear in monitor loop
- Set instruction → commentary streams + TTS audio plays in browser
- Switch to RTSP source → MJPEG video, TTS via `/api/audio-stream` fallback
- Verify OpenCV-only mode still works: `LIVEKIT_ENABLED=false python -m app.main`

---

## Step 7: Input Robustness and Testing

**Goal**: Reliable handling of all input sources with auto-reconnect. Test with real hardware.

**Files**:
- `app/frame_capture.py` -- Add reconnect logic with exponential backoff
- `app/config.py` -- Add `RECONNECT_DELAY`, `MAX_RECONNECT_DELAY`
- `app/main.py` -- Enhanced `/api/status` with `capture_healthy`, `tts_enabled`, `livekit_connected`

**Reconnect pattern**: On consecutive read failures (>10), release and re-open the VideoCapture. Exponential backoff from 1s to 30s. Reset delay on successful reconnect.

**Testing matrix** (manual, with user involvement):

| Source | Protocol | Tool | Test |
|--------|----------|------|------|
| Video file | File path | -- | Already works from Sprint 1 |
| VLC stream (same PC) | RTSP | `vlc --sout '#rtp{sdp=rtsp://:8554/stream}'` | Test URL input |
| VLC stream (other PC) | RTSP | Same but with LAN IP | Cross-machine test |
| IP webcam (old) | RTSP or HTTP/MJPEG | Depends on camera model | User provides URL |
| Phone camera | RTSP | IP Webcam app (Android) | RTSP URL from app |
| Browser webcam | WebRTC/LiveKit | Chrome/Firefox | LiveKit flow |

**Verification**:
- Each source type produces frames in the SlidingWindow
- Disconnecting and reconnecting a stream → auto-recovery
- `/api/status` accurately reports connection health
- Error messages are clear and actionable

---

## Step 8: UI Updates

**Goal**: Update web UI for MJPEG video, LiveKit webcam, TTS controls, status indicators.

**Depends on**: All backend steps

**Files**:
- `app/static/index.html` -- Major update (still vanilla HTML/JS/CSS, React in Sprint 3)

**UI additions**:
1. **Source mode selector**: Radio buttons -- "File/RTSP/Device" (OpenCV) vs "Browser Webcam" (LiveKit)
2. **MJPEG video**: `<img src="/api/mjpeg">` for OpenCV sources (smooth, replaces polling)
3. **LiveKit player**: livekit-client connects, shows local preview + receives TTS audio
4. **TTS controls**: Mute/unmute button, volume slider
5. **Status indicators**: Color-coded dots for capture health, LiveKit connection, model status, TTS status
6. **Source audio** (nice-to-have): If source is a LiveKit stream with audio, play it optionally (controlled via separate mute button)

**Verification**:
- OpenCV mode: MJPEG smooth video + SSE text + TTS audio via `<audio>` element
- LiveKit mode: WebRTC video + LiveKit TTS audio + SSE text
- Controls work: mute, volume, source switch
- Status indicators reflect actual state
- Graceful degradation: without LiveKit server, OpenCV mode works fine

---

## Implementation Order & Dependencies

```
Step 1: AWQ Model ─────────────────────────────────────────────┐
Step 2: MJPEG Stream ────────────────────────────────────────┐ │
Step 3: TTS Integration (depends on Step 1 for VRAM) ───────┤ │
Step 4: Audio Delivery (depends on Step 3) ─────────────────┤ │
Step 5: Docker (parallel with Steps 3-4) ───────────────────┤ │
Step 6: LiveKit (depends on Steps 4, 5) ────────────────────┤ │
Step 7: Input Robustness (depends on Step 6) ───────────────┤ │
Step 8: UI Updates (depends on Steps 2, 6) ─────────────────┘ │
                                                               │
Test & verify after each step ─────────────────────────────────┘
```

Steps 1 and 2 are independent quick wins. Steps 3-4 are the core TTS work. Step 5 can run in parallel. Steps 6-8 build on everything prior.

## Config Additions Summary

All new entries in `app/config.py`, env var overridable:

```python
# TTS
ENABLE_TTS = os.getenv("ENABLE_TTS", "false").lower() == "true"
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", "models/MiniCPM-o-4_5/assets/system_ref_audio.wav")
TTS_FLOAT16 = os.getenv("TTS_FLOAT16", "true").lower() == "true"

# LiveKit
LIVEKIT_ENABLED = os.getenv("LIVEKIT_ENABLED", "false").lower() == "true"
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret" * 6)

# Reconnect
RECONNECT_DELAY = float(os.getenv("RECONNECT_DELAY", "1.0"))
MAX_RECONNECT_DELAY = float(os.getenv("MAX_RECONNECT_DELAY", "30.0"))
```

## New Files Summary

| File | Purpose |
|------|---------|
| `app/audio_manager.py` | Audio resampling, pub/sub for audio chunks |
| `app/frame_provider.py` | Abstract FrameProvider + OpenCV/LiveKit implementations |
| `app/livekit_bot.py` | LiveKit room bot: video subscribe, audio publish |
| `Dockerfile` | App container (CUDA + Python) |
| `docker-compose.yml` | App + LiveKit server orchestration |
| `.env.example` | Documented config defaults |
| `config/livekit.yaml` | LiveKit server configuration |
| `scripts/test_awq.py` | AWQ model quality/VRAM test |
| `scripts/test_tts.py` | TTS quality/latency test |

## Risks

| Risk | Mitigation |
|------|------------|
| AWQ vision quality degraded | Test first (Step 1). Keep BF16 as fallback via MODEL_PATH. |
| TTS streaming API mismatch with our monitor loop | Simplex API is simple (prefill + generate loop). Test in isolation first (test_tts.py). |
| AWQ + TTS doesn't fit in 24 GB | AWQ=11 GB + TTS=2 GB = 13 GB. Should be fine. If not, disable TTS or reduce FRAMES_PER_INFERENCE. |
| LiveKit complexity escalates | Steps 1-5 deliver value without LiveKit. It's additive, not blocking. |
| Reference audio quality affects TTS | Model ships with `system_ref_audio.wav`. Can try alternatives from assets/. |
| autoawq custom fork breaks | Pin version. BF16 fallback always available. |

## Graceful Degradation

Everything is opt-in:
- `ENABLE_TTS=false` → text-only, same as Sprint 1
- `LIVEKIT_ENABLED=false` → OpenCV + MJPEG + SSE, no WebRTC needed
- `MODEL_PATH=models/MiniCPM-o-4_5` → BF16 fallback
- Any combination works. Full stack = AWQ + TTS + LiveKit + Docker.
