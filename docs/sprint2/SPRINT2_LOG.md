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

## Step 3: TTS Integration

### Research findings

Before writing any code, the TTS code paths in both model variants were traced using the repo-researcher agent. This revealed critical differences between the AWQ and BF16 model files.

**TTS architecture — two separate components:**

1. **TTS neural network** ("the brain"): A LlamaModel-based decoder that converts text into audio tokens. 194 weight tensors (`tts.*`) stored in the AWQ model's safetensors files. These are NOT quantized — they run in native precision even in the AWQ model. Loaded automatically when `init_tts: true` is set in `config.json` (which it is by default). No extra VRAM cost — already included in the ~8.6 GB.

2. **Token2wav vocoder** ("the voice"): A separate model (`stepaudio2.Token2wav`) that converts audio tokens into actual waveforms. Loaded from `assets/token2wav/` (~1.2 GB on disk). With `enable_float16=True`, uses ~0.6-0.7 GB VRAM.

**Critical AWQ vs BF16 difference in `init_tts()`:**

| Feature | AWQ model | BF16 model |
|---------|-----------|------------|
| `init_tts()` default vocoder | CosyVoice2 (incompatible with streaming) | Token2wav (works with streaming) |
| `streaming` parameter | Required (`streaming=True`) | Does not exist (always Token2wav) |
| `assets/` directory | **Missing entirely** | Present + auto-download |
| Auto-download of assets | No | Yes (`_ensure_asset_dir()`) |

The AWQ model's `modeling_minicpmo.py` is an older version of the code. Without `streaming=True`, it loads CosyVoice2, which crashes when `streaming_generate(generate_audio=True)` is called.

**Correct initialization for AWQ:**

```python
# Vocoder — MUST use streaming=True for AWQ
# Assets copied into AWQ dir (see docs/model_patches.md #4)
model.init_tts(
    streaming=True,
    model_dir="models/MiniCPM-o-4_5-awq/assets/token2wav",
    enable_float16=False,  # True crashes due to stepaudio2 dtype bug
)

# Reference audio — 16kHz mono numpy array
ref_audio, _ = librosa.load("models/MiniCPM-o-4_5-awq/assets/HT_ref_audio.wav", sr=16000, mono=True)
model.init_token2wav_cache(prompt_speech_16k=ref_audio)
```

**Streaming API (simplex, not duplex):**

```python
# Per inference cycle:
model.streaming_prefill(session_id=sid, msgs=[user_msg], is_last_chunk=True)
for wav_chunk, text_chunk in model.streaming_generate(
    session_id=sid, generate_audio=True, use_tts_template=True
):
    # wav_chunk: torch.Tensor (1, ~24000), 24kHz float32, ~1 second per chunk
    # text_chunk: str (incremental)
    # Final iteration yields (None, None) as sentinel
```

**VRAM budget with TTS:**

| Component | VRAM |
|-----------|------|
| AWQ model (LLM + vision + TTS weights) | ~8.6 GB |
| Token2wav vocoder (float32, float16 broken) | ~1.2 GB |
| KV cache during inference | ~2-3 GB |
| **Total measured** | **~12.1 GB allocated** |

~12 GB headroom on RTX 4090.

**Dependencies:** `minicpmo-utils[all]>=1.0.5` (provides `stepaudio2.Token2wav`), `torchaudio`, `librosa`.

**Simplex vs duplex:** Simplex (`streaming_prefill` + `streaming_generate`) is correct for our use case. Duplex (`model.as_duplex()`) adds VAD, listen/speak state machine, and continuous audio streaming — overkill for our pattern of frames + text in, text + audio out.

### Implementation

**Assets setup:** Copied `assets/` directory from BF16 model into AWQ directory (~1.3 GB). This makes the AWQ model self-contained — no dependency on BF16 model being present. Documented in `docs/model_patches.md` as patch #4.

**Code changes:**

- `app/config.py` -- Added `ENABLE_TTS` (default false), `TTS_MODEL_DIR`, `REF_AUDIO_PATH`, `TTS_FLOAT16` (default false due to bug)
- `app/model_server.py` -- Added `InferenceResult` dataclass, `_init_tts()` method (AWQ-aware vocoder init + ref audio cache), `infer_with_audio()` method using `streaming_prefill`/`streaming_generate`. Text-only `infer()` kept for backward compatibility.
- `app/requirements.txt` -- Added `minicpmo-utils[all]>=1.0.5`, `torchaudio`
- New: `scripts/test_tts.py` -- Test script: load with TTS, inference on image/video, save audio to WAV

**Key design:** `ModelServer` has two inference methods:
- `infer()` — text-only via `model.chat()` (existing, backward compatible)
- `infer_with_audio()` — yields `InferenceResult(text, audio, is_last)` via `streaming_prefill`/`streaming_generate`. Falls back to wrapping `infer()` when TTS disabled.

### Bugs found and fixed

**Bug 1 — float16 vocoder dtype mismatch:**
`enable_float16=True` causes `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half` in `stepaudio2.flow.setup_cache()`. The flow model's `spk_embed_affine_layer` has float16 weights but receives float32 speaker embeddings from the campplus ONNX extractor. This is a bug in stepaudio2, not our code.

Fix: changed `TTS_FLOAT16` default to `false`. Uses ~1.2 GB VRAM instead of ~0.6 GB. Acceptable on RTX 4090.

**Bug 2 — minicpmo-utils overrides torch version:**
`pip install minicpmo-utils[all]` pulled in torch 2.10.0, overriding our pinned torch==2.7.1. Also downgraded pillow and librosa to incompatible versions.

Fix: reinstalled pinned versions (`torch==2.7.1`, `torchvision==0.22.1`, `torchaudio==2.7.1`, `pillow==11.3.0`, `librosa==0.11.0`). The version conflict warnings from pip are harmless — stepaudio2 works fine with our versions.

### Test results

| Metric | Text-only | TTS (detailed prompt) | TTS (short prompt) |
|--------|-----------|----------------------|-------------------|
| VRAM (allocated) | ~8.6 GB | ~12.1 GB | ~12.1 GB |
| Inference time | ~1.6s | ~39s (870 chars) | ~5.0s (112 chars) |
| Audio output | N/A | 60.4s | 6.4s |
| Text quality | Good | Good | Good |

The ~5s inference with short prompt (1-2 sentences) is what we expect in the monitor loop with the commentator prompt.

### Config (all env-var overridable)

| Setting | Default | Purpose |
|---------|---------|---------|
| `ENABLE_TTS` | false | Enable TTS audio output |
| `TTS_MODEL_DIR` | `{MODEL_PATH}/assets/token2wav` | Path to Token2wav vocoder |
| `REF_AUDIO_PATH` | `{MODEL_PATH}/assets/HT_ref_audio.wav` | Reference audio for voice |
| `TTS_FLOAT16` | false | Float16 vocoder (currently broken, see Bug 1) |

---

## Step 4: Audio Delivery Pipeline

### Goal

Make TTS audio available to consumers via an HTTP endpoint. Step 3 produces audio; Step 4 delivers it.

### Implementation

**New file: `app/audio_manager.py`**

Pub/sub pattern for audio delivery (same pattern as MonitorLoop's text pub/sub):
- Receives 24kHz float32 torch tensors from `infer_with_audio()`
- Resamples to 48kHz int16 PCM using `scipy.signal.resample_poly` (WebRTC/playback standard)
- Distributes to subscriber queues (API endpoint, future LiveKit bot)
- Stop signal propagation for clean shutdown

Static method `resample_to_48k_int16(audio_24k: torch.Tensor) -> bytes` handles the conversion.

**Modified: `app/monitor_loop.py`**

`_inference_worker()` now branches on `self._model.tts_enabled`:
- TTS enabled: calls `infer_with_audio()`, publishes text chunks to text subscribers AND audio chunks (via AudioManager) to audio subscribers
- TTS disabled: calls `infer()` as before (no behavioral change)

`MonitorLoop.__init__()` accepts optional `AudioManager` parameter.

**Modified: `app/main.py`**

- AudioManager created in lifespan (only when `ENABLE_TTS=true`)
- Passed to MonitorLoop constructor
- Shutdown calls `audio_manager.stop()` to signal all audio subscribers
- `StatusResponse` includes `tts_enabled: bool`
- New endpoint: `GET /api/audio-stream` — raw PCM stream (48kHz, mono, int16 LE)
  - Returns 404 when TTS not enabled
  - Headers include format metadata (`X-Audio-Rate`, `X-Audio-Channels`, `X-Audio-Format`)
  - Subscriber-based: each connection gets its own queue (independent consumers)
  - 15s keepalive timeout for idle connections

### Code changes

- New: `app/audio_manager.py` — AudioManager class with resample + pub/sub
- `app/monitor_loop.py` — AudioManager integration in constructor and `_inference_worker()`
- `app/main.py` — AudioManager in lifespan, `/api/audio-stream` endpoint, `tts_enabled` in StatusResponse

### Test results

**Unit tests (AudioManager):**
- Resample: 24000 samples at 24kHz → 48000 samples at 48kHz (96000 bytes) — correct
- Pub/sub: subscribe → publish → receive → unsubscribe — correct
- Stop signal: `stop()` delivers `None` to all subscribers — correct

**Integration tests (server with TTS disabled):**
- `/api/status` returns `tts_enabled: false` — OK
- `/api/audio-stream` returns 404 — OK
- All existing endpoints unaffected — OK
- Server starts and shuts down cleanly — OK

**API usage:**
```bash
# Stream raw PCM audio (when TTS enabled)
curl http://localhost:8199/api/audio-stream > audio.pcm
ffplay -f s16le -ar 48000 -ac 1 audio.pcm

# Or pipe directly
curl -s http://localhost:8199/api/audio-stream | ffplay -f s16le -ar 48000 -ac 1 -
```

### Browser audio playback

Added Web Audio API player to `app/static/index.html`:
- Fetches `/api/audio-stream` as a streaming response via `fetch()` + `ReadableStream`
- Converts int16 PCM chunks to float32 for Web Audio API
- Schedules chunks as `AudioBufferSourceNode` for gapless playback
- Speaker button in header: mute/unmute toggle, hidden when TTS disabled
- "TTS" badge in header indicates TTS is active

### Live test results

Tested with `ENABLE_TTS=true`, video file, commentator prompt:
- **Audio delivery works**: TTS audio plays in browser via Web Audio API
- **Text streaming works**: SSE text commentary unaffected, still real-time
- **Mute/unmute works**: speaker button toggles audio stream on/off
- **TTS disabled mode**: speaker button hidden, everything works as before

### Resolved: audio-commentary timing

Initially, the timing between cycles was off — audio queued up and drifted behind real-time. This was resolved by the audio pacing implementation (see Step 4b below).

---

## Step 4b: Audio-Commentary Pacing

### Problem

Step 4's live test revealed three issues:
1. **Queue buildup (drift)**: Each cycle produced ~7s audio per ~6s real time, causing ~1s/cycle drift
2. **Uniform density**: Same commentary length regardless of scene activity
3. **No breathing room**: Cycles transitioned immediately with no pause
4. **Chinese audio on "..."**: Token2wav vocoder produced Chinese speech artifacts when the model output the skip signal "..."

### Implementation (per PLAN_audio_pacing.md)

**Step A — Audio-gated pacing:** AudioManager tracks `first_publish_time` + cumulative `audio_seconds`. MonitorLoop waits until `estimated_playback_end` before starting next cycle. Drift is eliminated by design.

**Step B — Breathing pause:** `TTS_PAUSE_AFTER` (default 1.0s) added after audio gate. Combined into a single wait: `remaining_audio + pause`. Configurable, 0 = no pause.

**Step C — Token cap:** `TTS_MAX_NEW_TOKENS` (default 150) limits TTS response length. ~2-3 sentences, ~8-12s audio. Prevents runaway 30s+ responses.

**Step D — Scene-weighted density:** `_scene_changed()` refactored to `_scene_diff()` returning a float score. New `_commentary_intensity()` uses dual-signal heuristic (pixel diff + previous response length) to vary prompt hints: minimal / brief / normal.

**"..." audio fix:** Audio chunks are buffered until accumulated text exceeds 5 chars. If the response turns out to be "...", the buffer is discarded (no audio published). Real responses flush the buffer and switch to real-time streaming with minimal delay.

### Code changes

- `app/audio_manager.py` — Clock tracking: `first_publish_time`, `audio_seconds`, `estimated_playback_end`, `reset_clock()`
- `app/monitor_loop.py` — Audio gate + pause after `_run_cycle()`, `_scene_diff()`, `_commentary_intensity()`, prompt hints, "..." audio suppression buffer
- `app/config.py` — Added `TTS_PAUSE_AFTER` (1.0), `TTS_MAX_NEW_TOKENS` (150)
- `app/model_server.py` — Uses `TTS_MAX_NEW_TOKENS` in `infer_with_audio()`

### Test results

Tested with full football match (Brazil vs France), `ENABLE_TTS=true`:
- **Drift**: None observed over extended playback. Audio gate holds sync.
- **Pacing**: Natural rhythm — more commentary during action, quieter during slow buildup.
- **Breathing pause**: Audible silence between observations. Feels natural.
- **"..." suppression**: No more Chinese audio artifacts during quiet moments.
- **Scoreboard reading**: Model reads match clock and score. Sometimes reads the time at end of a sentence, which by playback is a few seconds behind — cosmetic only, prompting can mitigate.

### Config

| Setting | Default | Purpose |
|---------|---------|---------|
| `TTS_PAUSE_AFTER` | 1.0 | Seconds of silence after audio before next cycle |
| `TTS_MAX_NEW_TOKENS` | 150 | Max tokens per TTS inference (shorter = shorter audio) |

See [Tuning Guide](../tuning_guide.md) for per-GPU recommendations and prompt tips.

---
