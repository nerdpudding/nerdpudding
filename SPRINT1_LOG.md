# Sprint 1 Log

Progress log for Sprint 1 MVP implementation. Documents setup steps, test results, and findings so everything is reproducible.

## Step 1: Environment Setup

### Conda environment

```bash
conda create -n video_chat python=3.12 -y
conda activate video_chat
pip install -r app/requirements.txt
```

Python 3.12 chosen because the model was built for `transformers==4.51.0` (see `config.json: "transformers_version": "4.51.0"`). Originally tried 4.55.0 from the CookBook requirements but it had breaking API changes (`DynamicCache.seen_tokens` removed, `generate()` return type changed).

### Model download

```bash
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir models/MiniCPM-o-4_5
```

- Full BF16, ~19 GB total (4 safetensor files + model code + TTS assets)
- Downloaded to `models/` inside the project (not `~/models/`), for Docker bind-mount compatibility later
- Model includes Python files (custom architecture, `trust_remote_code=True` required)
- Security scan of `.py` files: clean, no network calls during inference, no data exfiltration

### Model patch required

The downloaded model has a bug: `model.chat(stream=True)` crashes because `chat()` doesn't short-circuit for streaming and falls through to TTS post-processing that expects non-streaming output.

**Fix:** One-line patch in `models/MiniCPM-o-4_5/modeling_minicpmo.py` (after line ~1195).

See [docs/model_patches.md](docs/model_patches.md) for full details and how to verify/reapply.

### HuggingFace cache

`HF_HOME` is set to `models/.hf_cache/` (inside the project) to avoid polluting `~/.cache/huggingface/`. This is done in `app/model_server.py` before importing transformers.

### Verification

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
# True 12.6 NVIDIA GeForce RTX 4090
```

---

## Step 2: Model Server

### Files created

- `app/__init__.py` -- package marker
- `app/config.py` -- all configuration, environment variable overridable
- `app/model_server.py` -- model loading (vision-only) + `infer()` function
- `scripts/test_model.py` -- standalone test script

### Key design decisions

- Vision-only mode: `init_audio=False, init_tts=False, init_vision=True` (saves ~2-4 GB VRAM)
- `CUDA_VISIBLE_DEVICES` set in app code (not conda env), configurable via env var
- `HF_HOME` redirected to project-local `models/.hf_cache/`
- Streaming inference via `TextIteratorStreamer` (background thread in model code)
- `<|im_end|>` tokens filtered from output

### Test result

```bash
python -m scripts.test_model --image test_files/images/test.jpg
```

| Metric | Value |
|--------|-------|
| Model load time | 5.7s |
| Inference time (1 frame, full response) | 11.5s |
| Response length | 1759 chars |
| VRAM allocated | 16.4 GB |
| VRAM reserved | 16.7 GB |
| VRAM free | ~7.5 GB (of 24 GB) |

The model correctly described a Raspberry Pi robot car from the test image, with detailed component identification. Streaming worked -- text chunks arrived progressively.

### Performance notes

- 11.5s was for a long, unconstrained response (~512 tokens). The monitoring loop targets shorter updates with fewer `max_new_tokens`, so cycle time should be lower.
- 7.5 GB VRAM headroom is sufficient for multi-frame context (8 frames at 64 tokens/frame = 512 tokens).
- `transformers==4.51.0` shows a deprecation warning about `seen_tokens` -- harmless, will go away when/if the model code is updated upstream.

---
