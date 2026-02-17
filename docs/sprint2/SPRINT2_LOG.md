# Sprint 2 Log

Progress log for Sprint 2 implementation. Documents steps, test results, bugs found, and performance data.

## Step 1: AWQ Model Support

### Download

```bash
huggingface-cli download openbmb/MiniCPM-o-4_5-awq --local-dir models/MiniCPM-o-4_5-awq
```

- Official pre-quantized AWQ INT4 model from OpenBMB
- ~8 GB on disk (2 safetensor files + model code + tokenizer)
- Dependency added: `autoawq` from custom fork (`git+https://github.com/tc-mb/AutoAWQ.git`)

### Bugs found and fixed

**Bug 1 — AWQ config `modules_to_not_convert: null`:**
The published model's `config.json` has `"modules_to_not_convert": null`, which tells transformers to AWQ-quantize ALL linear layers. But only the LLM layers (`llm.model.layers.*`) were actually quantized in the safetensor files. The vision encoder's `intermediate_size: 4304` is not divisible by AWQ `group_size: 128`, causing `AssertionError` on load.

Fix: patched `config.json` with the correct exclusion list (9 module prefixes). See [docs/model_patches.md](../model_patches.md#2-awq-config-fix-modules_to_not_convert).

**Bug 2 — Same streaming bug as BF16:**
The AWQ model ships with the same unpatched `modeling_minicpmo.py`. Applied the same `chat()` streaming fix as the BF16 model.

**Bug 3 — HF cache serves stale model code:**
After patching `modeling_minicpmo.py` in `models/`, the HF cache in `models/.hf_cache/modules/` still served the old unpatched version. Had to delete the cache directory for the patched code to take effect.

### Code changes

- `app/model_server.py` -- Auto-detect AWQ via `hasattr(config, 'quantization_config')`, set `torch.float16` for AWQ, `torch.bfloat16` for BF16. Log model type on load.
- `app/config.py` -- Default `MODEL_PATH` changed to `models/MiniCPM-o-4_5-awq`.
- `app/requirements.txt` -- Added `autoawq` from custom fork.

### Test results

| Metric | AWQ INT4 | BF16 | Change |
|--------|----------|------|--------|
| VRAM (nvidia-smi) | ~8.6 GB | ~18.5 GB | -54% |
| Load time | 6.4s | 6.4s | same |
| Inference (single image, detailed) | 26.1s | 8.9s | slower* |
| Output quality | Good | Good | comparable |

*AWQ inference was slower on the single-image detailed test because the model generated a much longer response (2619 chars vs 1366 chars). In live commentary with the commentator prompt constraining length, inference times are comparable (~1.5-2.5s both).

### Switching models

```bash
# AWQ (default)
python -m app.main

# BF16 fallback
MODEL_PATH=models/MiniCPM-o-4_5 python -m app.main
```

---

## Step 1b: Latency Optimization

### Problem

End-to-end latency was ~10 seconds per cycle. Breakdown:
- Frame capture window: 8s (8 frames at 1 FPS)
- Inference: ~2s
- Idle wait (INFERENCE_INTERVAL): 5s between cycles

AWQ did not improve inference speed — the vision encoder (unquantized) dominates processing time.

### Solution: frame striding + tuned defaults

**Frame striding**: New `FRAME_STRIDE` config. Instead of N consecutive frames, take every Kth frame from the buffer. With stride=2 and n=4, we get 4 frames spanning double the time window, with half the image tokens sent to the model.

**Config changes** (new defaults):

| Setting | Before | After | Effect |
|---------|--------|-------|--------|
| `FRAMES_PER_INFERENCE` | 8 | 4 | Half the frames per cycle |
| `CAPTURE_FPS` | 1.0 | 2.0 | Frames 0.5s apart instead of 1s |
| `FRAME_STRIDE` | (new) | 2 | Every other frame, doubles temporal span |
| `INFERENCE_INTERVAL` | 5.0 | 1.0 | Minimal idle between cycles |

### Code changes

- `app/config.py` -- Added `FRAME_STRIDE`, updated defaults for `FRAMES_PER_INFERENCE`, `CAPTURE_FPS`, `INFERENCE_INTERVAL`
- `app/sliding_window.py` -- Added `stride` parameter to `get_frames()` and `get_frames_with_meta()`
- `app/monitor_loop.py` -- Passes `FRAME_STRIDE` when reading frames

### Test results

Tested with Shrek video file, AWQ model, commentator prompt active:

| Metric | Before (Sprint 1) | After (Step 1b) | Change |
|--------|-------------------|-----------------|--------|
| Avg latency | ~10s | ~4.8s | -52% |
| Avg inference | ~2.3s | ~1.6s | -30% |
| Cycles/minute | ~6 | ~12 | +100% |
| Frame span | 8s | 3s | -63% |
| Min latency | ~8.8s | ~3.5s | -60% |
| Commentary quality | Good | Good | comparable |

All settings remain env-var overridable for tuning per use case.

---

## Step 2: MJPEG Streaming with Adaptive Sync

### Problem

The Sprint 1 UI used frame polling (`setInterval` every 500ms fetching `/api/frame`). This produced a choppy slideshow effect. Additionally, the live video was always real-time while commentary referred to frames from several seconds ago, creating a disconnect.

### Solution: native-rate MJPEG + adaptive delay

**Two key changes:**

1. **Display/inference separation**: `FrameCapture` now reads every frame at native rate (~24 FPS for video files) and stores JPEG-encoded frames in a display buffer. It only pushes to `SlidingWindow` at `CAPTURE_FPS` (2 FPS) for inference. Previously, capture only ran at 2 FPS and the MJPEG could only show 2 frames/second.

2. **Adaptive sync delay**: The MJPEG endpoint serves frames from `now - target_delay` instead of the latest frame. `target_delay` is an EMA (Exponential Moving Average) that tracks observed inference latency. Skip cycles ("...") are filtered from the EMA to prevent drift.

### Research

Professional solutions for video-commentary sync were evaluated:
- [WeSpeakSports case study](https://ireplay.tv/blog/ultra-low-latency-webrtc-live-sports-commentary-wespeaksports-antmedia-mediasoup-altcasting/) — dual WebRTC for syncing live sports commentary
- [Adaptive Jitter Buffer](https://github.com/yingwang/adaptive-jitter-buffer) — playout speed based on buffer fullness
- [Fujimoto et al.](https://link.springer.com/article/10.1023/B:TELS.0000014784.20034.74) — EMA-based adaptive playout delay
- [Kalman et al. (Stanford)](https://web.stanford.edu/~bgirod/pdfs/KalmanCSVT2004.pdf) — adaptive media playout for low latency

Our approach (server-side EMA on observed latency) matches the academic consensus while being much simpler than a full jitter buffer — we have no network jitter, only variable inference time.

### Code changes

- `app/frame_capture.py` -- Major refactor: reads at native rate, dual output paths (display buffer at native FPS, inference callback at CAPTURE_FPS). New `get_display_jpeg(target_time)` method. JPEG-encodes every frame into a timestamped ring buffer (~30-45 MB for 15s at 24 FPS).
- `app/main.py` -- New `GET /api/mjpeg` endpoint using `StreamingResponse` with `multipart/x-mixed-replace`. Reads from display buffer with adaptive delay. Matches source FPS for smooth playback.
- `app/monitor_loop.py` -- Added `target_delay` property (EMA-updated on each non-skip `cycle_end`). Imported `STREAM_DELAY_INIT` and `STREAM_DELAY_EMA_ALPHA`.
- `app/sliding_window.py` -- Added `get_frame_near(target_timestamp)` method.
- `app/config.py` -- Added `STREAM_DELAY_INIT` (5.0s), `STREAM_DELAY_EMA_ALPHA` (0.2), `MJPEG_FPS` (10, fallback for live sources). Increased `WINDOW_SIZE` from 16 to 32.
- `app/static/index.html` -- Replaced `setInterval` frame polling with `<img src="/api/mjpeg">`. Shows `target_delay` in cycle metadata.

### Bugs found and fixed

**Bug 1 — Choppy MJPEG at 2 FPS:**
Initial implementation read from the SlidingWindow (2 FPS inference buffer). Video was as choppy as polling. Fixed by separating display from inference: frame_capture now reads at native rate (~24 FPS) and stores JPEG frames in a dedicated display buffer.

**Bug 2 — Skip cycles pulling EMA down:**
"..." responses (scene unchanged) have very short inference time (~0.3s), pulling the EMA delay too low. Fixed by filtering skipped cycles from the EMA update.

### Test results

| Metric | Before (polling) | After (MJPEG + sync) |
|--------|------------------|---------------------|
| Video frame rate | ~2 FPS (polling) | ~24 FPS (native) |
| Video smoothness | Choppy slideshow | Smooth playback |
| Video-commentary sync | No sync (live video, delayed text) | Adaptive sync via EMA |
| /api/frame (API) | Real-time | Still real-time (unchanged) |
| Display buffer memory | N/A | ~30-45 MB (15s at 24 FPS) |

EMA convergence example (from test run):
```
Cycle 1: inference=3.44s, target_delay=4.79s (from 5.0 init)
Cycle 2: inference=2.19s, target_delay=4.36s
Cycle 4: inference=2.09s, target_delay=3.99s  (skip cycles filtered)
Cycle 6: inference=1.86s, target_delay=3.64s
Cycle 7: inference=1.64s, target_delay=3.29s
...stabilizes around observed average
```

### Config (all env-var overridable)

| Setting | Default | Purpose |
|---------|---------|---------|
| `STREAM_DELAY_INIT` | 5.0 | Initial delay before EMA calibrates. 0 = no sync |
| `STREAM_DELAY_EMA_ALPHA` | 0.2 | EMA smoothing. Higher = faster adaptation |
| `MJPEG_FPS` | 10 | Fallback display FPS for live sources without known FPS |
| `WINDOW_SIZE` | 32 | Increased from 16 for delay headroom |

### Milestone: PoC Complete

With Steps 1, 1b, and 2 done, the project has a successful proof of concept: smooth native-rate video playback with real-time AI commentary, adaptively synchronized, running on consumer GPU hardware (~8.6 GB VRAM). All subsequent steps (TTS, LiveKit, Docker, etc.) are enhancements to an already working system.

---
