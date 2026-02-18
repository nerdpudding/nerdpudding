# Tuning Test Results

## Test 1: Beast mode (2026-02-18)

**Combo:** Beast (GPU: Push it / Tune: Preset D)
**Settings:** SLICE=2, 10 frames/cycle, stride=2, 5 FPS capture, CHANGE_THRESHOLD=0
**Image tokens/cycle:** 10 * 128 = 1280
**Source:** HTTP MJPEG webcam stream (localhost:8088), default commentator profile
**Instruction:** "Describe any movement or action..."
**TTS:** disabled

### Results (16 cycles)

| Metric | Min | Max | Typical |
|--------|-----|-----|---------|
| Inference time | 5.28s | 10.07s | 6-8s |
| Latency | 9.13s | 14.76s | 10-13s |
| Sync delay (EMA) | 9.7s | 13.6s | ~12s |

### Observations

1. **Too slow for interactive use.** Inference consistently above 5s threshold. At 6-8s per cycle, commentary lags 10+ seconds behind the action. Not viable for real-time.

2. **Object carry-over / hallucination.** The model repeatedly called different objects "container" — first correctly identified a container, then kept using that label for unrelated items (e.g., a cup being called "blue container"). The `_build_prompt` includes the previous response for context, which may anchor the model to prior labels.

3. **Repetitive commentary.** Many cycles produced variations of "adjusts posture" or "remains seated." Despite the "do not repeat" instruction, the model struggled to find new things to say when little changed.

4. **CHANGE_THRESHOLD=0 was necessary.** Earlier testing with threshold=2.0 caused most cycles to skip ("scene unchanged") because a webcam scene where 90% is static background produces very low mean pixel difference even with movement in the remaining 10%. Threshold=0 (never skip) was the only way to ensure responsiveness.

5. **Sync delay escalated.** The EMA-based adaptive delay started at 5.0s and climbed to 12-13s as it tracked the high inference times. This means the MJPEG stream was showing video from ~12 seconds ago to match commentary timing.

### Verdict

Beast is too heavy for the RTX 4090 with this model (AWQ INT4). 1280 image tokens per cycle pushes inference to 5-10s. Scale back to **Sniper** (6 frames, stride=1, 384 tokens/cycle) for the next test.

### Key takeaway

With MiniCPM-o 4.5 AWQ on RTX 4090:
- **< 640 image tokens/cycle** needed to stay under 5s inference
- SLICE=2 is viable but must reduce frame count to compensate
- Sniper (6 frames * 128 tokens = 768 tokens) is the next test point

---

## Test 2: Sniper (2026-02-18)

**Combo:** Sniper (GPU: Push it / Tune: Preset B)
**Settings:** SLICE=2, 6 frames/cycle, stride=1, 5 FPS capture, CHANGE_THRESHOLD=0
**Image tokens/cycle:** 6 * 128 = 768
**Source:** HTTP MJPEG webcam stream (localhost:8088), default commentator profile
**Instruction:** "Describe any movement or action..."
**TTS:** disabled

### Results (7 cycles, #21-#27)

| Metric | Min | Max | Typical |
|--------|-----|-----|---------|
| Inference time | 3.75s | 7.3s | 4-5.5s |
| Latency | 5.18s | 8.5s | 6-7s |
| Sync delay (EMA) | 4.77s | 5.4s | ~5.2s |

### Observations

1. **~35% faster inference than Beast.** Typical cycle 4-5.5s vs 6-8s. Some cycles dip below 5s but not consistently.

2. **Sync delay stable.** Stayed between 4.8-5.4s instead of climbing to 12+. Video-commentary sync works as designed.

3. **Latency roughly halved.** 6-7s typical vs 10-13s with Beast. Still not real-time but much more usable.

4. **Object recognition weak.** Model described movement ("hand moves", "extends arm", "gesture") but did not name objects being held or picked up. Likely caused by the instruction focusing on "movement or action" rather than object identification. Updated instruction for next test.

5. **Outlier at 7.3s.** Cycle 24 produced a longer two-sentence response. More output tokens = more decode time. Response length directly impacts cycle time.

### Verdict

Better than Beast but inference still not consistently under 5s. Two changes for next test:
- **Reduce FRAMES_PER_INFERENCE 6→4** (512 tokens instead of 768, should push inference under 4s)
- **Update instruction** to explicitly ask for object identification

### Key takeaway

768 image tokens/cycle is borderline on RTX 4090 with AWQ INT4. Need to get under ~512 for consistent sub-5s inference.

---

## Test 3: Sniper tuned — 4 frames + object instruction (2026-02-18)

**Combo:** Sniper tuned (GPU: Push it / Tune: Preset B modified)
**Settings:** SLICE=2, 4 frames/cycle, stride=1, 5 FPS capture, CHANGE_THRESHOLD=0
**Image tokens/cycle:** 4 * 128 = 512
**Source:** HTTP MJPEG webcam stream (localhost:8088), default commentator profile
**Instruction:** "Name every object the person touches, holds, or picks up. Describe what they do with it."
**TTS:** disabled

### Results (18 cycles)

| Metric | Min | Max | Typical |
|--------|-----|-----|---------|
| Inference time | 1.43s | 4.8s | 2.5-4.2s |
| Latency | 2.19s | 5.62s | 3.5-5.0s |
| Sync delay (EMA) | 3.61s | 5.0s | ~4.0s (trending down) |

### Comparison across all tests

| Combo | Tokens | Inference (typ) | Latency (typ) | Sync delay |
|-------|--------|-----------------|---------------|------------|
| Beast | 1280 | 6-8s | 10-13s | climbed to 12s |
| Sniper (6fr) | 768 | 4-5.5s | 6-7s | stable ~5.2s |
| **Sniper (4fr)** | **512** | **2.5-4.2s** | **3.5-5.0s** | **trending to ~4s** |

### Observations

1. **Best inference times yet.** Typical 2.5-4.2s, consistently under 5s. The "..." skip responses are very fast (1.4-2.5s). Real commentary responses 2.6-4.6s.

2. **Object recognition improved.** Model now identifies: pliers (cycle 4), clear glass (8), black smartphone (12), yellow object (14-15), black remote/controller (16), cardigan (17). Much better than the generic "hand moves" from test 2.

3. **Still misses fast transitions.** When objects are quickly swapped, the model often catches the second object but misses the first. The 0.8s time window means each cycle only sees 4 consecutive frames — fast hand movements can fall between cycles.

4. **"..." skips are appropriate.** Cycles 1-3, 5-7 correctly returned "..." when nothing was being held. The model respects the instruction scope.

5. **Sync delay trending down.** Started at 5.0s, settled toward 3.6-4.0s. Lower inference times pull the EMA down, which means less video delay.

6. **Longer responses = slower cycles.** Cycle 16 (4.52s, two-clause sentence) vs cycle 14 (2.64s, short phrase). Output token count still matters.

### Verdict

This is the sweet spot for the current hardware. 512 image tokens keeps inference under 5s while SLICE=2 provides enough detail for object recognition. The updated instruction dramatically improved object identification.

### Remaining issues
- Fast object swaps between cycles still get missed
- Color accuracy varies ("yellow object" — was it actually yellow?)
- Generic fallbacks like "black object, possibly a remote or controller" — hedging instead of committing

### Key takeaway

**512 image tokens (4 frames * SLICE=2) is the sweet spot on RTX 4090 AWQ INT4.**
- Inference consistently 2.5-4.5s
- Instruction wording matters as much as model parameters for output quality
