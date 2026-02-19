# Tuning Guide

All settings are in `app/config.py` and overridable via environment variables. No code changes needed — just set the variable before starting the server.

```bash
SETTING_NAME=value python -m app.main
```

Or use a `.env` file with your preferred settings.

## Quick Reference

Settings are grouped in `app/config.py` into **GPU presets** (detail level) and **tune presets** (capture/pacing). The active combo determines what values you get out of the box. See [Presets](#presets) below for all combinations.

| Setting | Range | Purpose |
|---------|-------|---------|
| `ENABLE_TTS` | true/false | Enable text-to-speech audio output |
| `TTS_MAX_NEW_TOKENS` | 64-512 | Max response length with TTS (tokens) |
| `TTS_PAUSE_AFTER` | 0-5.0 | Silence between commentary segments (seconds) |
| `MAX_NEW_TOKENS` | 128-1024 | Max response length without TTS (tokens) |
| `INFERENCE_INTERVAL` | 0.5-10.0 | Min pause between inference cycles (seconds) |
| `CHANGE_THRESHOLD` | 0-50 | Scene change sensitivity (pixel diff, 0=never skip) |
| `FRAMES_PER_INFERENCE` | 1-8 | Frames sent per cycle (more = more context, slower) |
| `FRAME_STRIDE` | 1-4 | Skip every Nth frame (higher = wider time span) |
| `CAPTURE_FPS` | 0.5-5.0 | Inference capture rate (not display rate) |
| `MAX_SLICE_NUMS` | 1-9 | Image detail level (1=fast, higher=detailed+slow) |
| `STREAM_DELAY_INIT` | 0-15.0 | Initial video-commentary sync delay (0=no sync) |
| `MODEL_PATH` | path | Model directory (default: `models/MiniCPM-o-4_5-awq`) |
| `SERVER_HOST` | IP address | Bind address (default: `127.0.0.1`, use `0.0.0.0` for network/Docker) |
| `SERVER_PORT` | port number | Server port (default: `8199`) |

## Presets

`config.py` organizes tuning into two layers that combine into a **combo**:

1. **GPU preset** — controls image detail level (`MAX_SLICE_NUMS`, `MAX_INP_LENGTH`)
2. **Tune preset** — controls frame capture, pacing, and responsiveness

To switch: uncomment the desired preset block in `config.py` and restart. Or override individual settings via environment variables without editing the file.

### GPU presets

| Setting | Conservative | Push it | Maximum |
|---------|-------------|---------|---------|
| `MAX_SLICE_NUMS` | 1 (64 tok/frame) | 2 (128 tok/frame) | 3 (192 tok/frame) |
| `MAX_INP_LENGTH` | 4352 | 8192 | 8192 |
| Best for | 10-12 GB GPUs | RTX 4090 (24 GB) | Only if SLICE=2 misses detail |

### Tune presets

| Setting | A: Sentry | B: Sniper | C: Owl | D: Beast |
|---------|-----------|-----------|--------|----------|
| `CAPTURE_FPS` | 2.0 | 5.0 | 4.0 | 5.0 |
| `FRAMES_PER_INFERENCE` | 4 | 4 | 8 | 10 |
| `FRAME_STRIDE` | 2 | 1 | 2 | 2 |
| `CHANGE_THRESHOLD` | 5.0 | 0 | 1.0 | 0 |
| `INFERENCE_INTERVAL` | 1.0 | 0.5 | 0.5 | 0.5 |
| `TTS_MAX_NEW_TOKENS` | 150 | 96 | 120 | 96 |
| `TTS_PAUSE_AFTER` | 1.0 | 0.5 | 0.5 | 0.3 |
| Time window | 4s | 0.8s | 4s | 4s |
| Image tokens (SLICE=2) | 512 | 512 | 1024 | 1280 |
| Intended for | Slow scenes, security | Fast action, webcam | Broad context, meetings | Max input, powerful GPU |

### Combos (GPU + Tune)

| Combo | GPU preset | Tune preset | Image tokens/cycle |
|-------|-----------|-------------|-------------------|
| Sentry | Conservative | A: Sentry | 256 |
| Sniper Lite | Conservative | B: Sniper | 256 |
| **Sniper** | **Push it** | **B: Sniper** | **512** |
| Owl | Push it | C: Owl | 1024 |
| Beast | Push it | D: Beast | 1280 |

### Important: presets are starting points, not optimized configurations

These presets have **not been systematically validated** across different scenarios. They provide reasonable starting points based on the hardware constraints, but the actual values may need adjustment depending on your specific use case, video content, and preferences.

What we know from testing (see [Tuning Test Results](tuning_test_results.md) for full data):

- **512 image tokens/cycle** (Sniper + Push it) is the sweet spot on RTX 4090 AWQ — inference consistently 2.5-4.5s
- **1280 tokens** (Beast + Push it) is too slow on RTX 4090 — inference 6-8s, latency 10-13s. May work on faster hardware.
- **768 tokens** (6 frames at SLICE=2) is borderline — inference 4-5.5s, inconsistently under the 5s target
- Instruction wording has as much impact on output quality as parameter tuning
- Systematic tuning with controlled test plans is on the [roadmap](../roadmap.md)

Tuning involves many interacting factors (frame count, detail level, threshold, TTS length, prompting, video content type). Expect to experiment.

## Video Sources

The system accepts any source that OpenCV's `VideoCapture` can open. Enter the source in the browser UI's "Video source" field.

| Source | Format | Example |
|--------|--------|---------|
| Local video file | File path | `/home/user/match.mp4` |
| Webcam | Device ID (integer) | `0` |
| RTSP stream | RTSP URL | `rtsp://192.168.1.100:554/stream` |
| HTTP MJPEG stream | HTTP URL | `http://192.168.1.100:8080/video` |
| HTTP video stream | HTTP URL | `http://example.com/stream.mp4` |

Video files loop automatically — useful for testing and development.

### Tips for specific sources

**Phone as camera (Android/iOS):**
Install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) (Android) or a similar app. It serves an MJPEG stream on your local network. Use the URL it shows (e.g. `http://192.168.1.50:8080/video`).

**VLC re-streaming:**
Stream any content from another PC as RTSP:
```bash
vlc input.mp4 --sout '#rtp{sdp=rtsp://:8554/stream}'
```
Then use `rtsp://<that-pc-ip>:8554/stream` as the source.

**IP cameras:**
Most IP cameras serve RTSP or HTTP MJPEG. Check your camera's manual for the stream URL. Common patterns:
- `rtsp://<ip>:554/stream1` (many brands)
- `http://<ip>/cgi-bin/mjpeg` (older cameras)
- `rtsp://user:password@<ip>:554/Streaming/Channels/1` (Hikvision)

**YouTube / Twitch:**
Not supported directly (DRM, dynamic URLs). Use `yt-dlp -g <url>` to extract the direct stream URL. Results vary by format.

### Scene detection by source type

Different sources produce different levels of visual change. Adjust `CHANGE_THRESHOLD` accordingly.

**Important:** These suggestions are rough guidelines. Testing showed that webcam scenes with a mostly static background (person at desk) produce very low mean pixel difference even with movement — threshold values like 3.0-5.0 caused most cycles to skip. Set to `0` (never skip) when in doubt, then increase if the AI talks too much about nothing.

| Source type | Suggested starting point | Notes |
|---|---|---|
| Sports broadcast | 5.0-8.0 | Frequent camera cuts produce high diff |
| Animation / gaming | 3.0-5.0 | Consistent visual changes |
| Security camera (static) | 8.0-15.0 | Skip lighting changes, react to movement |
| Live webcam (person at desk) | 0-2.0 | Mostly static background suppresses diff; start at 0 |
| Phone camera (handheld) | 0-2.0 | Constant hand movement; threshold may fight with noise |

## VRAM Usage

| Mode | Approximate VRAM | Notes |
|------|-----------------|-------|
| Text-only (AWQ) | ~8.6 GB | Default, no TTS |
| Text + TTS (AWQ) | ~14-15 GB | ENABLE_TTS=true |
| Text-only (BF16) | ~18.5 GB | MODEL_PATH=models/MiniCPM-o-4_5 |

VRAM includes model weights, KV cache during inference, and vocoder (when TTS enabled). Peak usage occurs during inference — idle usage is lower.

## Tuning for Your GPU

### RTX 4090 (24 GB) — reference hardware

Default settings work well. Plenty of headroom for TTS.

```bash
ENABLE_TTS=true python -m app.main
```

### RTX 3090 / 4080 (16 GB)

TTS fits but with less headroom. Consider shorter responses:

```bash
ENABLE_TTS=true TTS_MAX_NEW_TOKENS=96 python -m app.main
```

### RTX 3080 / 4070 Ti (12 GB)

TTS may be tight. Text-only is safe. If attempting TTS, reduce everything:

```bash
ENABLE_TTS=true TTS_MAX_NEW_TOKENS=64 FRAMES_PER_INFERENCE=2 MAX_SLICE_NUMS=1 python -m app.main
```

If you get CUDA out-of-memory errors, disable TTS and use text-only mode.

### RTX 3060 / 4060 (8 GB)

Text-only with AWQ model only. TTS will not fit.

```bash
python -m app.main
```

## TTS Pacing

When TTS is enabled, the system uses audio-gated pacing: the next inference cycle waits until the current audio finishes playing, then adds a configurable pause.

**If commentary feels too rushed:**
```bash
TTS_PAUSE_AFTER=2.0   # More silence between segments
TTS_MAX_NEW_TOKENS=96  # Shorter responses
```

**If commentary feels too slow:**
```bash
TTS_PAUSE_AFTER=0.5    # Less silence
TTS_MAX_NEW_TOKENS=200 # Allow longer responses
```

**If commentary talks too much about nothing:**
```bash
CHANGE_THRESHOLD=10.0  # Require bigger scene changes before commenting
```

## Scene Change Detection

The system uses pixel difference between frames to detect scene changes. Below `CHANGE_THRESHOLD`, the cycle is skipped (no inference, no audio). Set to 0 to never skip (always run inference).

See [Scene detection by source type](#scene-detection-by-source-type) above for per-source recommendations.

## Video-Commentary Sync

The MJPEG stream shows video with an adaptive delay that matches the commentary timing. This keeps what you see aligned with what you hear.

- `STREAM_DELAY_INIT=5.0` — initial delay before the system calibrates (default)
- `STREAM_DELAY_INIT=0` — disable sync, show real-time video (commentary will refer to frames from a few seconds ago)

The system automatically adjusts the delay using an EMA (Exponential Moving Average) of observed inference latency. No tuning needed after the first few cycles.

## Prompt Tips

The instruction you type in the UI shapes the commentary style. Some examples:

**General commentary:**
```
Describe what you see. Focus on actions and changes.
```

**Sports commentator (adapt team names to your match):**
```
Commentate on this football match between Brazil (BRA) and France (FRA).
The scoreboard shows country abbreviations, the score, and the match clock
— the clock is NOT the score. Focus on exciting moments: attacks, shots,
saves, fouls, corners, and near-misses. Build tension during dangerous plays.
Be enthusiastic about goal chances, not monotone. Skip boring buildup in
midfield — only speak when something interesting happens. You may mention
the match time only at 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 60, 65,
70, 75, 80, 85, 90, or more than 90 minutes play time.
```

Without the time constraint, the model tends to read the clock in every sentence. Adding specific intervals makes it mention the time only occasionally — much more natural.

**Security monitoring:**
```
Monitor this camera feed. Only report when a person enters or leaves the frame,
or when something unusual happens. Stay silent otherwise.
```

**Nature / wildlife:**
```
Narrate this nature scene like a documentary. Describe animal behavior,
movements, and interactions. Speak softly and calmly.
```

Tips:
- Tell the model what to focus on AND what to ignore
- For TTS: shorter instructions tend to produce more concise audio
- For sports: explain scoreboard layout so the model reads it correctly
- Use "stay silent" / "only speak when" to reduce unnecessary commentary

The web UI also has a **prompt profile dropdown** with predefined system prompts (General, Sports, Security, Nature, Descriptive). These set the AI's personality — you can switch live without restarting. Profiles are defined in `app/config.py` under `PROMPT_PROFILES`.

## Test Results

For detailed benchmark data on different preset combinations and attention backends, see [Tuning Test Results](tuning_test_results.md). Key findings:

- Sniper (512 tokens, SLICE=2) is the best-performing combo on RTX 4090 AWQ: 2.5-4.5s inference
- Beast (1280 tokens) causes 6-8s inference — too slow for interactive use on this hardware
- SageAttention v1/v2 and Flash Attention 2 do not improve on PyTorch SDPA flash for this model
- `torch.compile()` provides a modest improvement on skip responses
