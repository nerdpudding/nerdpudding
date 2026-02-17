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
