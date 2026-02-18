# Sprint 2 Review

## Summary

Sprint 2 transformed the Sprint 1 MVP from a text-only slideshow into a full audio-visual commentary system. The AI now watches video at native frame rate, speaks its commentary aloud with natural pacing, and adapts its verbosity to match scene activity. Tested end-to-end with a full football match — the system produces a convincing live sports commentary experience.

**What was built:**
- AWQ INT4 model support (54% VRAM reduction: 18.5 GB to 8.6 GB)
- Frame striding for latency optimization (52% latency reduction: ~10s to ~4.8s)
- MJPEG streaming with adaptive video-commentary sync (EMA-based delay tracking)
- TTS integration via model's built-in streaming simplex API (Token2wav vocoder, 24kHz output)
- Audio delivery pipeline with browser playback (Web Audio API, 48kHz PCM streaming)
- Audio-commentary pacing system (audio gate, breathing pause, token cap, scene-weighted density)
- "..." audio suppression (prevents Chinese speech artifacts from Token2wav vocoder)
- Tuning guide with per-GPU recommendations, TTS pacing, and prompt tips

**What was deferred to Sprint 3:**
- Docker setup (GPU passthrough, docker-compose)
- LiveKit WebRTC (browser webcam input, TTS audio via WebRTC)
- Input robustness (RTSP/IP cam testing, auto-reconnect)
- UI updates (source mode selector, LiveKit player, status indicators)

These were originally Sprint 2 Steps 5-8 but are better as a separate sprint focused on deployment and input flexibility, now that the core pipeline is complete.

## Architecture

```
Video Source  --[native FPS]--> Frame Capture --[display buffer]--> MJPEG endpoint
(cam/stream/file)                    |                                 (adaptive delay)
                              [2 FPS push]
                                     v
                              Sliding Window
                                     |
                              [4 frames, stride 2]
                                     v
Web UI <--[SSE]-- FastAPI <--[pub/sub]-- Monitor Loop --[infer]--> Model Server
  |         |         ^                    |  ^                    (AWQ INT4, TTS)
  |    [PCM stream]   |              [audio gate]                       |
  |         v         |                    v                            |
  |    Web Audio API  |              Audio Manager <--- 24kHz float32 --+
  |    (48kHz int16)  |              (resample + pub/sub)
  |                   |     user questions
  +-------------------+
```

**Two milestones reached:**
1. **PoC Complete** (Steps 1-2): Smooth video + real-time text commentary + adaptive sync
2. **Sprint 2 Complete** (Steps 1-4b): Full pipeline with TTS audio and natural pacing

## Key Findings

### Performance

| Metric | Sprint 1 | Sprint 2 (text-only) | Sprint 2 (with TTS) |
|--------|----------|---------------------|---------------------|
| VRAM (nvidia-smi) | ~18.5 GB (BF16) | ~8.6 GB (AWQ) | ~14-15 GB (AWQ + vocoder + KV cache) |
| Inference per cycle | ~2.3s avg | ~1.6s avg | ~5s avg |
| End-to-end latency | ~10s | ~4.8s avg | Audio-gated (adaptive) |
| Display frame rate | ~2 FPS (polling) | ~24 FPS (MJPEG) | ~24 FPS (MJPEG) |
| Commentary output | Text (SSE) | Text (SSE) | Text (SSE) + audio (Web Audio API) |
| Cycles/minute | ~6 | ~12 | ~4-6 (audio-gated, varies by scene) |

### VRAM Budget

| Component | VRAM |
|-----------|------|
| AWQ INT4 model (LLM + vision + TTS weights) | ~8.6 GB |
| Token2wav vocoder (float32) | ~1.2 GB |
| KV cache during inference | ~2-3 GB |
| **Total peak (with TTS)** | **~14-15 GB** |
| Idle (no inference) | ~10 GB |

### Latency Breakdown (text-only mode)

| Component | Sprint 1 | Sprint 2 | Change |
|-----------|----------|----------|--------|
| Frame age | ~7s (8 frames, 1 FPS) | ~3s (4 frames, stride 2) | -57% |
| Inference | ~2.3s | ~1.6s (fewer image tokens) | -30% |
| Idle wait | 5s | 1s | -80% |
| **Total** | **~10s** | **~4.8s** | **-52%** |

### Model Observations

- AWQ INT4 produces comparable quality to BF16 for commentary — no noticeable degradation
- AWQ inference speed is similar to BF16 (vision encoder dominates, runs unquantized)
- TTS audio quality is good for commentary. Male English voice via reference audio.
- Token2wav vocoder produces Chinese speech artifacts on "..." tokens — requires audio suppression
- `enable_float16=True` for vocoder crashes due to dtype mismatch in stepaudio2 (bug in upstream library)
- Shorter prompts produce dramatically shorter inference times with TTS (~5s vs ~39s for detailed prompts)
- The model reads scoreboards and match clocks but tends to mention the time every sentence — constraining via prompt fixes this

### Architecture Observations

- Audio-gated pacing eliminates drift by design: measure instead of predict playback timing
- Dual-signal commentary intensity (pixel diff + previous response length) is more robust than pixel diff alone
- Audio buffering for "..." suppression adds negligible latency (~0.5s delay before first chunk streams)
- Web Audio API streaming with scheduled AudioBufferSourceNodes produces gapless playback
- EMA-based adaptive sync converges within 5-10 cycles and handles mode switches (TTS on/off, prompt changes)
- Separating display buffer (native FPS) from inference buffer (2 FPS) was critical — first attempt used a single buffer and produced 2 FPS MJPEG

## Bugs Fixed

1. **AWQ config `modules_to_not_convert: null`** — Published model's config.json told transformers to AWQ-convert all layers, but only LLM layers were quantized. Vision encoder's dimensions aren't divisible by AWQ group size, causing AssertionError. Fixed by patching config.json.

2. **AWQ streaming crash** — Same `chat()` streaming bug as BF16 model (falls through to TTS post-processing). Applied same one-line patch.

3. **HF cache serves stale model code** — After patching model Python files, cached (unpatched) versions were still loaded from HF cache. Fixed by clearing cache directory.

4. **Choppy MJPEG at 2 FPS** — Initial MJPEG read from SlidingWindow (2 FPS inference buffer). Fixed by adding a separate display buffer at native frame rate.

5. **Skip cycles pulling EMA down** — "..." responses (~0.3s inference) dragged EMA delay too low. Fixed by filtering skip cycles from EMA updates.

6. **float16 vocoder dtype mismatch** — `enable_float16=True` causes RuntimeError in stepaudio2's flow model. Fixed by defaulting to float32 (~0.6 GB more VRAM, acceptable).

7. **minicpmo-utils overrides torch version** — `pip install minicpmo-utils[all]` pulled incompatible torch version. Fixed by reinstalling pinned versions after.

8. **Chinese TTS audio on "..."** — Token2wav vocoder produced Chinese speech when given "..." tokens. Fixed by buffering audio until text exceeds 5 chars, discarding buffer for "..." responses.

9. **AWQ missing assets directory** — AWQ model doesn't ship with `assets/` directory (vocoder + reference audio). Fixed by copying from BF16 model. Documented as patch #4.

## Tuning Options

All tunable via environment variables. See [Tuning Guide](../tuning_guide.md) for full details.

### TTS Pacing

```bash
# Default TTS with good pacing
ENABLE_TTS=true python -m app.main

# Faster commentary (less pause between segments)
ENABLE_TTS=true TTS_PAUSE_AFTER=0.5 python -m app.main

# Shorter responses
ENABLE_TTS=true TTS_MAX_NEW_TOKENS=96 python -m app.main
```

### Video-Commentary Sync

```bash
# Adaptive sync (default, delay tracks inference latency)
python -m app.main

# No sync (real-time video, commentary refers to past)
STREAM_DELAY_INIT=0 python -m app.main
```

### Scene Change Sensitivity

```bash
# More sensitive (comment on smaller changes)
CHANGE_THRESHOLD=3.0 python -m app.main

# Less sensitive (skip minor changes, only comment on big shifts)
CHANGE_THRESHOLD=10.0 python -m app.main
```

## What Worked Well

- **Audio-gated pacing** — solved the drift problem completely on first implementation. Measuring playback time is much more reliable than predicting it.
- **External plan review** — sending implementation plans to another AI for review caught three substantive issues before implementation: timing startpoint correction, sample rate verification, and reconnect scenario. Worth the extra time.
- **Research before coding** — repo-researcher agent traced the TTS code paths (AWQ vs BF16 differences in `init_tts()`, simplex vs duplex API) before writing integration code. Saved several debugging iterations.
- **Incremental testing** — each step tested independently before moving on. Step 3 (TTS) bugs were isolated before Step 4 (delivery) added complexity.
- **Commentator prompt tuning** — the prompt makes a huge difference for TTS. Restricting when the model can mention match time (only at specific minute marks) made commentary feel much more natural.

## Limitations

- **No Docker** — still runs directly on host. Not reproducible for others without manual setup.
- **No browser webcam input** — only OpenCV sources (files, RTSP, device IDs). LiveKit deferred to Sprint 3.
- **No reconnection logic** — camera disconnect requires manual restart.
- **TTS float16 broken** — upstream stepaudio2 bug prevents float16 vocoder. Uses ~0.6 GB more VRAM than necessary.
- **Pixel diff heuristic** — camera pans produce high diff with nothing meaningful; small scoreboard changes produce low diff but are important. Dual-signal approach mitigates but doesn't solve.
- **Single browser session** — multiple browsers streaming audio can cause timing inconsistencies in the audio gate (server tracks one clock).
- **No persistent config** — environment variables must be set each run.

## Sprint 3 Recommendations

Based on Sprint 2 findings:

### High Priority
- **Docker setup** — containerize for reproducibility. Model as bind mount, docker-compose with GPU passthrough. This is the main barrier to others using the project.
- **LiveKit WebRTC** — browser webcam input + TTS audio back via WebRTC. Eliminates the need for external video sources for testing.

### Medium Priority
- **Input robustness** — auto-reconnect on source failure, status reporting, test matrix with real hardware (RTSP, IP cam, phone).
- **UI updates** — source mode selector (OpenCV vs LiveKit), status indicators, better TTS controls (volume slider).

### Considerations
- Docker and LiveKit can be developed in parallel — they're independent until integration.
- LiveKit adds complexity (Go server, JWT tokens, bot participant). Consider if simpler WebRTC (peer-to-peer) is sufficient.
- Test with multiple browsers before assuming single-session audio gate is adequate.

## Files Created/Modified in Sprint 2

| File | Purpose |
|------|---------|
| `app/audio_manager.py` | NEW: Audio resampling (24kHz→48kHz) + pub/sub + clock tracking for pacing |
| `app/model_server.py` | AWQ auto-detection, `InferenceResult` dataclass, `infer_with_audio()`, TTS init |
| `app/frame_capture.py` | Native-rate display buffer, dual output paths (display + inference) |
| `app/sliding_window.py` | Frame striding, `get_frame_near()` for delayed MJPEG |
| `app/monitor_loop.py` | Audio gate, scene diff/intensity, prompt hints, "..." suppression |
| `app/main.py` | MJPEG endpoint, audio-stream endpoint, AudioManager lifecycle |
| `app/static/index.html` | MJPEG video, Web Audio API player, speaker mute/unmute, TTS badge |
| `app/config.py` | TTS settings, sync settings, frame striding, tuning defaults |
| `app/requirements.txt` | autoawq, minicpmo-utils, torchaudio, scipy |
| `scripts/test_tts.py` | NEW: TTS quality/latency test script |
| `docs/tuning_guide.md` | NEW: Per-GPU settings, TTS pacing, prompt tips |
| `docs/model_patches.md` | AWQ config patch, assets copy documentation |
| `docs/sprint2/SPRINT2_LOG.md` | Step-by-step progress log with measurements |
| `archive/2026-02-19_PLAN_audio_pacing.md` | Audio pacing implementation plan (externally reviewed, archived) |
