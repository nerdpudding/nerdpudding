# Tuning Guide

All settings are in `app/config.py` and overridable via environment variables. No code changes needed — just set the variable before starting the server.

```bash
SETTING_NAME=value python -m app.main
```

Or use a `.env` file with your preferred settings.

## Quick Reference

| Setting | Default | Range | Purpose |
|---------|---------|-------|---------|
| `ENABLE_TTS` | false | true/false | Enable text-to-speech audio output |
| `TTS_MAX_NEW_TOKENS` | 150 | 64-512 | Max response length with TTS (tokens) |
| `TTS_PAUSE_AFTER` | 1.0 | 0-5.0 | Silence between commentary segments (seconds) |
| `MAX_NEW_TOKENS` | 512 | 128-1024 | Max response length without TTS (tokens) |
| `INFERENCE_INTERVAL` | 1.0 | 0.5-10.0 | Min pause between inference cycles (seconds) |
| `CHANGE_THRESHOLD` | 5.0 | 0-50 | Scene change sensitivity (pixel diff, 0-255) |
| `FRAMES_PER_INFERENCE` | 4 | 1-8 | Frames sent per cycle (more = more context, slower) |
| `FRAME_STRIDE` | 2 | 1-4 | Skip every Nth frame (higher = wider time span) |
| `CAPTURE_FPS` | 2.0 | 0.5-5.0 | Inference capture rate (not display rate) |
| `MAX_SLICE_NUMS` | 1 | 1-9 | Image detail level (1=fast, higher=detailed+slow) |
| `STREAM_DELAY_INIT` | 5.0 | 0-15.0 | Initial video-commentary sync delay (0=no sync) |
| `MODEL_PATH` | models/MiniCPM-o-4_5-awq | path | Model directory |

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

The system uses pixel difference between frames to detect scene changes. Below `CHANGE_THRESHOLD`, the cycle is skipped (no inference, no audio).

- **Animation / gaming**: 5.0 (default) works well
- **Live video with camera movement**: Try 3.0-5.0 (cameras create constant change)
- **Static security camera**: Try 8.0-15.0 (skip minor lighting changes)
- **Sports broadcast**: 5.0-8.0 (frequent cuts and camera pans)

Set to 0 to never skip (always run inference).

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
