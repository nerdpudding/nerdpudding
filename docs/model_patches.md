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

## 2. AWQ config fix: modules_to_not_convert

**File:** `models/MiniCPM-o-4_5-awq/config.json`
**Section:** `quantization_config`
**Model version:** `openbmb/MiniCPM-o-4_5-awq` downloaded 2026-02-17

### Problem

The published AWQ model has `"modules_to_not_convert": null` in `config.json`. This causes `AutoModel.from_pretrained()` to try to AWQ-quantize ALL linear layers, including the vision encoder (SiglipVisionTransformer) whose `intermediate_size: 4304` is not divisible by the AWQ `group_size: 128`. This crashes with `AssertionError: self.in_features % self.group_size == 0`.

In reality, only the LLM layers (`llm.model.layers.*`) are quantized in the safetensor files. Vision, audio, TTS, and embedding layers are stored as regular float weights.

### Fix

Changed `modules_to_not_convert` from `null` to the list of modules that should NOT be AWQ-converted:

```json
"modules_to_not_convert": ["vpm", "resampler", "apm", "audio_projection_layer", "audio_avg_pooler", "tts", "llm.model.embed_tokens", "llm.lm_head", "llm.model.norm"]
```

### How to verify the patch is applied

```bash
grep "modules_to_not_convert" models/MiniCPM-o-4_5-awq/config.json
```

Should show the list of modules, NOT `null`.

## 3. AWQ streaming fix in chat() method

**File:** `models/MiniCPM-o-4_5-awq/modeling_minicpmo.py`
**Line:** ~1198 (after `self.generate()` call, before TTS post-processing)
**Model version:** `openbmb/MiniCPM-o-4_5-awq` downloaded 2026-02-17

Same bug as patch #1, same fix. The AWQ model ships with the same unpatched `modeling_minicpmo.py`.

### How to verify the patch is applied

```bash
grep -n "PATCH: chat" models/MiniCPM-o-4_5-awq/modeling_minicpmo.py
```

Should show the patch comment around line 1198-1201.

## 4. TTS assets for AWQ model

**Directory:** `models/MiniCPM-o-4_5-awq/assets/`
**Source:** Copied from `models/MiniCPM-o-4_5/assets/`

### Problem

The AWQ model download does not include the `assets/` directory. This directory contains the Token2wav vocoder (~1.2 GB) needed for TTS audio generation, plus reference audio files for voice cloning.

Without these assets, `model.init_tts(streaming=True)` fails because it cannot find the vocoder model files (`flow.pt`, `hift.pt`, `speech_tokenizer_v2_25hz.onnx`, `campplus.onnx`).

### Fix

Copy the entire `assets/` directory from the BF16 model into the AWQ model directory:

```bash
# If BF16 model is already downloaded:
cp -r models/MiniCPM-o-4_5/assets models/MiniCPM-o-4_5-awq/assets

# Or download just the assets from HuggingFace:
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir models/MiniCPM-o-4_5 --include "assets/*"
cp -r models/MiniCPM-o-4_5/assets models/MiniCPM-o-4_5-awq/assets
```

### How to verify

```bash
ls models/MiniCPM-o-4_5-awq/assets/token2wav/
```

Should show: `campplus.onnx`, `flow.pt`, `flow.yaml`, `hift.pt`, `speech_tokenizer_v2_25hz.onnx`.
