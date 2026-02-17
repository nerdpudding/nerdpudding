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
