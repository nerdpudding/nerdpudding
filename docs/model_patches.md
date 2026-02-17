# Model Patches

Patches applied to the downloaded model files. These must be reapplied after re-downloading the model or updating to a new version.

## 1. Streaming fix in chat() method

**File:** `models/MiniCPM-o-4_5/modeling_minicpmo.py`
**Line:** ~1196 (after `self.generate()` call, before TTS post-processing)
**Model version:** `openbmb/MiniCPM-o-4_5` downloaded 2026-02-17

### Problem

`model.chat(stream=True)` crashes with `AttributeError: 'dict' object has no attribute 'sequences'`. When `stream=True`, `generate()` returns `(TextIteratorStreamer, {})` but `chat()` continues to TTS post-processing that tries `outputs.sequences[0]` on the empty dict.

The CookBook's `minicpmo4_5.py` wrapper has the same bug -- it catches the error with try/except and silently returns an error string.

### Fix

Added early return for streaming after `self.generate()`:

```python
        # PATCH: chat() doesn't handle stream=True -- it falls through to TTS
        # post-processing that crashes on outputs.sequences (empty dict).
        # Return the TextIteratorStreamer directly. See docs/model_patches.md.
        if stream:
            return res
```

This returns the `TextIteratorStreamer` directly from `chat()`, skipping the TTS post-processing which is irrelevant for streaming text output.

### How to verify the patch is applied

```bash
grep -n "PATCH: chat" models/MiniCPM-o-4_5/modeling_minicpmo.py
```

Should show the patch comment around line 1196-1199.

### Docker note

When building a Docker image, either:
- Apply this patch in the Dockerfile after copying the model files
- Or include the patched file in the image build context
