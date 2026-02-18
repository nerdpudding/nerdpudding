# Implementation Plan: Audio-Commentary Pacing

## Problem Statement

With TTS enabled, the inference cycle produces ~6-7s of audio per ~5s of inference time. The monitor loop starts the next cycle after a fixed 1s interval, causing:

1. **Queue buildup (drift)**: Each cycle adds ~1s more audio than the gap between cycles. After 10 cycles, the browser is ~10s behind real-time and falling further behind.
2. **Uniform density**: Commentary length is the same regardless of scene activity. Not natural.
3. **No breathing room**: Cycles transition immediately, never leaving silence between observations.

Text-only mode doesn't have these issues because text appears/disappears instantly.

## Design Principles

- Audio is a time-bound medium. The system must respect playback duration.
- A natural commentator varies pace: talks more during action, stays quiet during calm.
- Silence between observations is a feature, not a bug.
- The inverse scenario (inference takes longer than audio) creates natural "thinking pauses" — this is acceptable and intentional. Do NOT build workarounds for it.
- All new settings must be env-var overridable and only affect TTS mode.

## Implementation Steps

### Step A: Audio-gated pacing (fixes drift)

**Goal:** The monitor loop waits until the browser has (approximately) finished playing the previous cycle's audio before starting the next cycle.

**Files changed:**
- `app/audio_manager.py` — Add audio clock tracking
- `app/monitor_loop.py` — Use audio clock to gate next cycle

**AudioManager changes:**

Add three fields to track the audio clock:
```python
self._first_publish_time: Optional[float] = None  # time.time() of first chunk this cycle
self._audio_seconds: float = 0.0                   # cumulative audio duration this cycle
```

Add methods:
```python
# SAMPLE_RATE and BYTES_PER_SAMPLE as class constants for clarity:
SAMPLE_RATE = 48000   # after resample_to_48k_int16()
BYTES_PER_SAMPLE = 2  # int16

def publish(self, data: Optional[bytes]) -> None:
    """Send audio data to all subscribers. Also tracks audio clock.

    NOTE: publish() receives ALREADY RESAMPLED data (48kHz int16 PCM).
    The conversion chain is:
      Token2wav (24kHz float32 tensor)
      → AudioManager.resample_to_48k_int16() in monitor_loop._inference_worker()
      → publish(pcm_bytes)  ← this method
    So len(data) / (48000 * 2) correctly gives seconds of audio.
    """
    if data is not None:
        if self._first_publish_time is None:
            self._first_publish_time = time.time()
        self._audio_seconds += len(data) / (self.SAMPLE_RATE * self.BYTES_PER_SAMPLE)
    for q in self._subscribers:
        q.put_nowait(data)

def reset_clock(self) -> None:
    """Reset audio clock for new cycle. Call at cycle start."""
    self._first_publish_time = None
    self._audio_seconds = 0.0

@property
def estimated_playback_end(self) -> float:
    """Estimated wall-clock time when the browser finishes playing current audio.

    Returns 0.0 if no audio was published (no waiting needed).
    """
    if self._first_publish_time is None:
        return 0.0
    return self._first_publish_time + self._audio_seconds
```

**Why `first_publish_time` and not `cycle_start`:** Audio chunks stream during inference. The browser starts playing as soon as the first chunk arrives, which is after prefill (~0.5-1s into inference). Using `cycle_start` would overestimate the wait by that prefill duration. At 5s inference time, that's a significant error.

**MonitorLoop changes (`run()`):**

After `_run_cycle()` completes, before starting the next cycle, check the audio clock:

```python
# After _run_cycle returns:
if self._audio_manager is not None:
    playback_end = self._audio_manager.estimated_playback_end
    now = time.time()
    remaining = playback_end - now
    if remaining > 0:
        logger.info(f"Audio gate: waiting {remaining:.1f}s for playback to finish")
        await asyncio.sleep(remaining)
```

Then also add a `reset_clock()` call at the start of each cycle (in `_run_cycle`, before inference begins):
```python
if self._audio_manager is not None:
    self._audio_manager.reset_clock()
```

**What happens when there's no audio (skip cycle / "..."):**
`estimated_playback_end` returns 0.0, `remaining` is negative, no waiting. Next cycle starts immediately. This is correct.

**What happens when inference is longer than audio (short remark, long thinking):**
Audio finishes playing before inference finishes. By the time inference ends, `remaining` is negative. No waiting. The silence during inference IS the natural pause. This is correct and intentional.

### Step B: Inter-cycle breathing pause

**Goal:** Add a configurable silence after the audio finishes, before the next cycle starts. Real commentators don't go straight from one observation to the next.

**Files changed:**
- `app/config.py` — Add `TTS_PAUSE_AFTER` setting
- `app/monitor_loop.py` — Add pause after audio gate

**Config:**
```python
# Seconds of silence after TTS audio finishes before the next inference cycle.
# Creates natural breathing room between observations — real commentators pause.
# Only effective when ENABLE_TTS=true. Has no effect in text-only mode.
# 0 = no pause (next cycle starts as soon as audio finishes).
# 1.0 = good default for natural pacing. Increase on faster GPUs if commentary
# feels rushed. Decrease (or set to 0) on slower GPUs where inference time
# already provides enough gap between audio segments.
TTS_PAUSE_AFTER = float(os.getenv("TTS_PAUSE_AFTER", "1.0"))
```

**MonitorLoop change:**

After the audio gate wait (Step A), add the pause:
```python
if self._audio_manager is not None:
    playback_end = self._audio_manager.estimated_playback_end
    now = time.time()
    remaining = playback_end - now + TTS_PAUSE_AFTER  # include breathing pause
    if remaining > 0:
        logger.info(f"Audio gate: waiting {remaining:.1f}s (playback + pause)")
        await asyncio.sleep(remaining)
```

This is simpler than a separate sleep — combine the audio wait and the pause into one. If audio is already done, the pause alone applies. If audio is still playing, the total wait = remaining_audio + pause.

**Note:** We considered making the pause variable based on scene diff (less action = longer pause). This is deferred — a fixed pause is good enough for now and avoids overcomplicating this step. Can be revisited once the baseline pacing feels right.

### Step C: MAX_NEW_TOKENS cap for TTS mode (safety valve)

**Goal:** Prevent runaway-long responses that produce 30+ seconds of audio.

**Files changed:**
- `app/config.py` — Add `TTS_MAX_NEW_TOKENS` setting
- `app/model_server.py` — Use lower cap when TTS active

**Config:**
```python
# Max tokens per inference when TTS is active. Lower = shorter audio output.
# Controls the upper bound on how long each commentary segment can be.
# 150 tokens ≈ 2-3 sentences ≈ 8-12 seconds of audio. Good for natural pacing.
# Only effective when ENABLE_TTS=true. Text-only mode uses MAX_NEW_TOKENS (512).
# Set to 0 to use MAX_NEW_TOKENS for TTS too (not recommended — can produce
# 30+ seconds of audio per cycle, causing queue buildup).
# On slower GPUs, consider lowering to 96-128 for faster cycle times.
# On faster GPUs, 150-200 works well.
TTS_MAX_NEW_TOKENS = int(os.getenv("TTS_MAX_NEW_TOKENS", "150"))
```

**ModelServer change:**

In `infer_with_audio()`, use `TTS_MAX_NEW_TOKENS` if set:
```python
effective_max_tokens = TTS_MAX_NEW_TOKENS if TTS_MAX_NEW_TOKENS > 0 else MAX_NEW_TOKENS

# ... in streaming_generate call:
max_new_tokens=effective_max_tokens,
```

### Step D: Scene-weighted commentary length

**Goal:** Vary commentary density based on how much changed in the scene. Low change = brief. High change = more detail. This creates the organic rhythm.

**Files changed:**
- `app/monitor_loop.py` — Change `_scene_changed()` to return a score, use score in prompt with dual-signal heuristic

**_scene_changed() refactor:**

Currently returns `bool`. Change to return `float` (the diff score), and rename to `_scene_diff()`:
```python
def _scene_diff(self, current_frame: Image.Image) -> float:
    """Compute mean pixel difference from last inference frame.

    Returns 0.0 if no previous frame (first cycle).
    Returns float in range 0-255.
    """
    if self._last_inference_frame is None:
        return 255.0  # First frame = maximum "change"
    try:
        old = np.array(self._last_inference_frame.resize((64, 64)), dtype=np.float32)
        new = np.array(current_frame.resize((64, 64)), dtype=np.float32)
        return float(np.mean(np.abs(old - new)))
    except Exception:
        return 255.0
```

Callers update: `if diff < CHANGE_THRESHOLD: skip` (same logic as before).

**Dual-signal heuristic for length hints:**

Pixel diff alone is a weak proxy for "something interesting happened." A camera pan produces high diff with nothing noteworthy; a scoreboard change gives low diff but is important. To improve this, combine two signals:

1. **Pixel diff** (visual change) — already computed, fast, no extra inference cost
2. **Previous response length** (semantic signal) — if the model's last response was short or "...", it saw nothing interesting. If it was long, there was a lot to talk about.

The previous response is a lagging indicator (it reflects the LAST cycle, not this one), but it correlates well because adjacent frames tend to have similar activity levels. Together the two signals are more robust than either alone.

```python
def _commentary_intensity(self, scene_diff: float) -> str:
    """Determine commentary length hint based on scene diff and recent history.

    Returns a prompt fragment: 'minimal', 'brief', or 'normal'.
    """
    # Previous response length as secondary signal
    prev_len = len(self._last_response.strip()) if self._last_response else 0
    prev_was_short = prev_len < 40  # "..." is 3, a short phrase is ~20-40 chars

    if scene_diff < 15 and prev_was_short:
        return "minimal"  # Both signals agree: not much happening
    elif scene_diff < 15 or (scene_diff < 40 and prev_was_short):
        return "brief"    # One signal says calm
    else:
        return "normal"   # Enough change to describe freely
```

**_build_prompt() changes:**

```python
def _build_prompt(self, instruction: str, scene_diff: float) -> str:
    parts = [COMMENTATOR_PROMPT]
    if self._last_response and self._last_response.strip() != "...":
        parts.append(
            f'\nYour last comment was: "{self._last_response}"\n'
            "Do not repeat this. Only add new observations."
        )
    # Scene-weighted length hint (only for TTS mode)
    if self._model.tts_enabled:
        intensity = self._commentary_intensity(scene_diff)
        if intensity == "minimal":
            parts.append("\nVery little changed. Be extremely brief — one short phrase at most.")
        elif intensity == "brief":
            parts.append("\nSome things changed. Keep it to one sentence.")
        # "normal": no constraint — describe freely (still bounded by TTS_MAX_NEW_TOKENS)
    parts.append(f"\nFocus: {instruction}")
    return "\n".join(parts)
```

**Known limitation — pixel diff weaknesses:**

Pixel diff is a spatial heuristic that doesn't understand semantics. Known failure modes:
- **Camera pan over static scene**: High pixel diff, but nothing meaningful to comment on. The dual-signal approach mitigates this — if the model's previous response was short despite high diff, the next cycle biases toward "brief."
- **Small but significant change** (e.g. scoreboard update): Low pixel diff, but important. The CHANGE_THRESHOLD (5.0) may skip these entirely. For now this is acceptable — the model is a general commentator, not a scoreboard reader.
- **Lighting changes** (day/night transition, cloud shadows): Can cause gradual diff changes that trigger unnecessary commentary.

These could be improved later with structural similarity (SSIM) instead of mean pixel diff, or by analyzing the model's own output patterns over time. For now, the dual-signal approach (pixel diff + previous response length) is a pragmatic starting point that handles the most common cases.

## Implementation Order

1. **Step A** — Audio-gated pacing. This is the critical fix. Without it, the queue drifts regardless of other changes.
2. **Step B** — Inter-cycle pause. Quick addition on top of A. Together they create the basic "natural" feeling.
3. **Step C** — Token cap. Quick safety valve.
4. **Step D** — Scene-weighted prompts. Most subjective, needs tuning. Do last so A+B+C are stable.

Test after each step:
- A: Verify no audio drift over 20+ cycles. Verify text-only mode unaffected.
- B: Verify pause is audible between cycles. Verify skip cycles have no pause.
- C: Verify long responses are capped. Verify text-only MAX_NEW_TOKENS unchanged.
- D: Verify commentary is shorter on static scenes, longer on active scenes.

## Files Summary

| File | Step | Change |
|------|------|--------|
| `app/audio_manager.py` | A | Add clock tracking (first_publish_time, audio_seconds, estimated_playback_end, reset_clock) |
| `app/monitor_loop.py` | A, B, D | Audio gate wait + pause after _run_cycle, _scene_diff() refactor, dual-signal _commentary_intensity(), prompt length hints |
| `app/config.py` | B, C | Add TTS_PAUSE_AFTER, TTS_MAX_NEW_TOKENS |
| `app/model_server.py` | C | Use TTS_MAX_NEW_TOKENS in infer_with_audio() |

No changes to: `main.py`, `index.html`, `audio_manager.py` (beyond Step A).

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Server-side audio clock is inaccurate vs actual browser playback | Audio gate waits too long or too short | Clock tracks first_publish_time (when data hits the network), not cycle_start. Small network delay is absorbed by TTS_PAUSE_AFTER. |
| Scene diff thresholds in Step D are wrong for some content | Commentary too brief or too verbose for certain videos | Dual-signal heuristic (pixel diff + previous response length) is more robust than pixel diff alone. Thresholds can be tuned after testing. Known limitation: camera pans, small important changes. See Step D notes. |
| TTS_MAX_NEW_TOKENS too low, cuts off mid-sentence | Jarring audio cutoff | 150 tokens is ~2-3 sentences, enough for natural commentary. Model tends to end sentences before token limit. |
| Audio gate + pause makes the system feel sluggish | Commentary updates too slowly | TTS_PAUSE_AFTER is configurable (0-N seconds). Can be reduced or set to 0. Audio gate alone is exact — no unnecessary waiting. |
| Browser disconnects/reconnects mid-cycle | Server-side clock assumes audio was played; after reconnect the estimate is stale | Non-critical: clock resets each cycle via reset_clock(), so stale state only persists for one cycle. After reconnect, the next cycle_start resets the clock and everything re-syncs. Worst case is one cycle with slightly wrong timing. |

## Acceptable Behavior (by design, not bugs)

- **Silence during long inference**: If inference takes 5s but only produces 2s audio, there's 3s of silence while the model "thinks". This is natural commentator behavior — observing before speaking.
- **Variable cycle rate**: Cycles are no longer on a fixed cadence. Active scenes trigger rapid short cycles; calm scenes have long gaps. This is the goal.
- **Skip cycles still have no audio**: "..." responses produce no audio. The audio gate sees 0 duration and proceeds immediately (plus the breathing pause if configured).
