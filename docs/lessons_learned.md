# Lessons Learned

Ongoing log of what worked and what didn't during development. Primarily intended as context for AI assistants to avoid repeating mistakes, but useful for anyone picking up the project.

---

## Research before coding

**Lesson:** Thoroughly investigate how external code works before writing integration code. Don't assume APIs based on method signatures or documentation alone -- trace the actual code paths.

**Example (Sprint 1, Step 2):** When implementing streaming inference, multiple iterations failed because the model's `chat()` method was assumed to support `stream=True` based on it accepting the parameter. In reality, `chat()` had a bug where it never short-circuited for streaming and crashed on TTS post-processing. This was only discovered after several failed attempts. A repo-researcher agent traced the full code path (`chat()` -> `generate()` -> `_decode_stream()`) and identified the exact bug in one pass. That research should have happened first, before writing any code.

**Rule:** When integrating with external code (model libraries, APIs, demos), use the repo-researcher agent to trace the full call chain before writing integration code. Don't iterate by trial-and-error.

---

## Pin dependency versions to what the model expects

**Lesson:** Check `config.json` in the model directory for `transformers_version` before choosing dependency versions. The model's custom Python code is written for a specific transformers API.

**Example (Sprint 1, Step 2):** The CookBook's `requirements.txt` pinned `transformers==4.55.0`, but the model's `config.json` specified `"transformers_version": "4.51.0"`. Using 4.55.0 caused `DynamicCache.seen_tokens` errors because the API changed between versions. Downgrading to 4.51.0 (what the model was built for) fixed it.

**Rule:** Always check the model's `config.json` for `transformers_version` and use that version, not what a demo or cookbook pins.

---

## Keep caches inside the project

**Lesson:** HuggingFace caches model code to `~/.cache/huggingface/` by default when using `trust_remote_code=True`. This causes issues with reproducibility and Docker deployments.

**Example (Sprint 1, Step 2):** Stale cached model code from an earlier transformers version persisted in `~/.cache/` and was loaded instead of the local model files. Setting `HF_HOME` to `models/.hf_cache/` before importing transformers fixed this and keeps everything project-local.

**Rule:** Set `HF_HOME` to a project-local path before importing transformers. Do it at module level, before the import statement, not inside a function that runs later.

---

## asyncio.create_task does not execute immediately

**Lesson:** `asyncio.create_task(coro)` schedules the coroutine but does not run it until the current coroutine yields. Code that depends on the task having started (checking flags, calling methods) will see stale state.

**Example (Sprint 1, Step 5):** The monitor loop was started with `create_task(monitor.run())`, but consumers that checked `monitor._running` immediately after saw `False` and exited -- producing no output. Also, calling `stop()` before `run()` actually started caused `run()` to override `_running` back to `True`, creating an infinite loop.

**Fix:** Added an `asyncio.Event` (`_started`) that `run()` sets when it begins executing, and a `wait_started()` method for consumers to await. Also added `_stop_requested` flag to prevent `run()` from starting after `stop()`.

**Rule:** When starting async tasks that other code depends on, use an Event or similar synchronization primitive. Never rely on flags set inside the task being visible immediately after `create_task()`.

---

## Commentator-style prompting produces better live commentary

**Lesson:** Open-ended prompts like "describe what you see" produce long, repetitive responses that are slow and hard to follow. A structured system prompt with explicit rules dramatically improves quality for live commentary.

**Example (Sprint 1, Step 8):** Switching from bare user instructions to a commentator system prompt (1-2 sentences, change-only focus, no repetition, "..." for unchanged scenes) cut inference time from 7-9s to 1.2-2.3s per cycle and made responses much more useful. Adding context carry-over ("your last comment was: ...") prevented the model from repeating itself across cycles.

**Rule:** For continuous monitoring use cases, always use a system prompt that constrains response length and focuses on changes. Include the previous response in the prompt to prevent repetition.

---

## Centralize all configuration from the start

**Lesson:** Hardcoded values scattered across multiple Python files make it difficult to tune the system and easy to miss when adjusting parameters.

**Example (Sprint 1, Step 8):** After building the full pipeline, values like JPEG quality, suppress tokens, and the commentator prompt were hardcoded in various files. Moving everything to a single `config.py` with env var overrides made it possible to tune latency, response length, and behavior without editing code.

**Rule:** Create a central config file at the start of a project. Every parameter that might need tuning should be there from day one, overridable via environment variables.

---

## Video files read sequentially in OpenCV, not by time

**Lesson:** `cv2.VideoCapture.read()` returns the next frame in sequence for video files, regardless of elapsed time. At 25 FPS source and 1 FPS capture rate, reading once per second gives you frames 0, 1, 2... (covering 0.04s each), not frames spread across the video.

**Example (Sprint 1, Step 3):** The capture thread read one frame per second, but 5 reads in 5 seconds only covered the first 0.2 seconds of a 25 FPS video.

**Fix:** For video files, skip `src_fps / capture_fps` frames per interval using `grab()` (fast, no decode) before `read()` (decode only the target frame). Not needed for live sources where `read()` returns the latest frame.

**Rule:** When capturing from video files at a lower rate than the source FPS, use `grab()` to skip intermediate frames. Only call `retrieve()`/`read()` on the frame you actually want.

---

## AWQ models may ship with broken quantization configs

**Lesson:** Pre-quantized AWQ models from HuggingFace may have `"modules_to_not_convert": null` in `config.json`, even though only a subset of layers were actually quantized. This causes `AutoModel.from_pretrained()` to try to AWQ-convert all linear layers, crashing on layers whose dimensions aren't divisible by the AWQ group size.

**Example (Sprint 2, Step 1):** The `openbmb/MiniCPM-o-4_5-awq` model's `config.json` had `"modules_to_not_convert": null`. Only the LLM layers were quantized (verified via safetensor weight inspection), but the vision encoder's `intermediate_size: 4304` is not divisible by `group_size: 128`, causing `AssertionError` during loading. The CookBook documentation recommends using `AutoAWQForCausalLM.from_quantized()` from a forked AutoAWQ library, which handles this internally. Alternatively, patching `config.json` with the correct `modules_to_not_convert` list fixes it for `AutoModel.from_pretrained()`.

**Rule:** When loading AWQ models, always check if `modules_to_not_convert` is set correctly in `config.json`. If null, inspect the safetensor index to determine which modules were actually quantized, and set the list manually. Also clear the HF cache (`models/.hf_cache/modules/`) after patching model code, or the stale cached version gets loaded.

---

## Attention backend alternatives don't help this model

**Lesson:** PyTorch SDPA flash (built into PyTorch 2.7) is already near-optimal for MiniCPM-o 4.5 AWQ on RTX 4090. Alternative attention backends either don't help or actively break things.

**Tested (Sprint 3):**
- SageAttention v1 (INT8, Triton JIT): ~50% slower than SDPA flash
- SageAttention v2 (INT4+FP8, compiled CUDA): CUDA kernel crash in Qwen3 LLM attention
- Flash Attention 2 (Dao-AILab, compiled from source): no measurable difference vs SDPA flash
- torch.compile(): modest improvement on skip responses, kept

**Rule:** Don't assume "quantized attention" or "specialized kernels" = faster. Always benchmark against the PyTorch default. For short sequences (~700 tokens) with AWQ models, SDPA flash is hard to beat. See `docs/tuning_test_results.md` for full data.

---

## Clear HF cache after patching model code

**Lesson:** When using `trust_remote_code=True`, transformers caches the model's Python files in `HF_HOME/modules/transformers_modules/<model_name>/`. If you patch the model's `.py` files in the `models/` directory, the cached (unpatched) version may still be loaded.

**Example (Sprint 2, Step 1):** After patching `models/MiniCPM-o-4_5-awq/modeling_minicpmo.py` for the streaming fix, the test still crashed because transformers loaded the cached unpatched version from `models/.hf_cache/modules/transformers_modules/MiniCPM-o-4_5-awq/`.

**Rule:** After patching any model `.py` file, always delete the corresponding cache directory: `rm -rf models/.hf_cache/modules/transformers_modules/<model_name>/`. The cache will be rebuilt from the patched source on next load.

---
