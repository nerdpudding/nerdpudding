# Plan: Attention Optimization (SageAttention)

## Context

NerdPudding runs MiniCPM-o 4.5 (AWQ INT4) on RTX 4090. Current Sniper preset:
- 4 frames/cycle, SLICE=2, 512 image tokens
- Inference: 2.5-4.5s per cycle
- Attention is ~30-40% of inference time

PyTorch SDPA with flash_sdp backend is already active (built into PyTorch 2.7).
The standalone `flash-attn` package gives only ~2-4% total improvement over this.

SageAttention quantizes the attention computation itself (orthogonal to AWQ which
quantizes linear layer weights). This is the biggest remaining optimization.

## Decision: SageAttention 2

**Why v2 over v1:**
- v1 (INT8): ~2.1x faster attention, ~15-20% total speedup
- v2 (INT4+FP8): ~3x faster attention, ~25-30% total speedup
- v2 has better accuracy than v1 (outlier smoothing, adaptive mixed precision)
- v2 is strictly better in both speed and accuracy
- Only reason to prefer v1 is "pip install" convenience — irrelevant in Docker

**Why not flash-attn:**
- SDPA flash is already active in PyTorch 2.7
- flash-attn package gives ~2-4% total improvement — not worth the dependency

**AWQ compatibility:** Confirmed. AWQ quantizes weights (linear layers), SageAttention
quantizes attention computation. They operate on different parts of the model and
are tested together (SageAttention paper includes AWQ benchmarks).

## Expected results

| Current | + SageAttention 2 |
|---------|-------------------|
| 2.5-4.5s inference | ~1.8-3.2s inference |
| 3.5-5.0s latency | ~2.5-4.0s latency |

## Implementation

### 1. Integration in model_server.py

One-line monkey-patch before model loading:

```python
# In model_server.py, before AutoModel.from_pretrained()
try:
    from sageattention import sageattn
    import torch.nn.functional as F
    F.scaled_dot_product_attention = sageattn
    logger.info("SageAttention enabled")
except ImportError:
    logger.info("SageAttention not installed, using PyTorch SDPA")
```

Keep `attn_implementation="sdpa"` — it calls `F.scaled_dot_product_attention`
internally, which now routes to SageAttention.

Graceful fallback: if sageattention is not installed, SDPA flash works as before.
No breakage, no config flag needed.

### 2. Docker build (ties into PLAN_sprint3_dockerization.md)

In Dockerfile, compile SageAttention from source with explicit GPU arch list:

```dockerfile
# Exclude Blackwell (sm_120) — known issues, not yet stable in SageAttention
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# SageAttention 2 from source
RUN git clone https://github.com/thu-ml/SageAttention.git /tmp/sageattention && \
    cd /tmp/sageattention && \
    pip install --no-build-isolation . && \
    rm -rf /tmp/sageattention
```

Supported GPUs with this arch list:
- 8.0 = A100, A800 (Ampere datacenter)
- 8.6 = RTX 3080, 3090 (Ampere consumer)
- 8.9 = RTX 4090, 4080, L40 (Ada)
- 9.0 = H100, H800 (Hopper datacenter)

Blackwell (sm_120) is excluded. When SageAttention stabilizes on Blackwell,
add "12.0" to the arch list.

### 3. Non-Docker / local install

For users who don't use Docker:

```bash
# Set arch list to avoid Blackwell compilation issues
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
pip install sageattention --no-build-isolation
```

Or for specific GPU only (faster compilation):
```bash
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090 only
pip install sageattention --no-build-isolation
```

### 4. README instructions

Add a section to README:

```
## Attention optimization (optional)

SageAttention is automatically used if installed. It speeds up inference by
~25-30% by quantizing the attention computation.

If you experience issues (wrong output, crashes), uninstall it:
    pip uninstall sageattention

The app will fall back to PyTorch's built-in attention automatically.

Known issue: SageAttention does not currently support Blackwell GPUs
(RTX 5070/5080/5090). If you only have a Blackwell GPU, skip this step.
```

### 5. Fallback chain (no config flags needed)

1. SageAttention installed → uses SageAttention (fastest)
2. SageAttention not installed → PyTorch SDPA flash (already fast)
3. SDPA flash not available → PyTorch SDPA memory-efficient (still fine)

The try/except in model_server.py handles this. No env vars, no config toggles.
Users who have issues just `pip uninstall sageattention`.

## Files to modify

| File | Change |
|------|--------|
| `app/model_server.py` | Add SageAttention monkey-patch (try/except) |
| `Dockerfile` | Add SageAttention compilation step |
| `requirements.txt` | Add `sageattention` (with comment: optional, built in Docker) |
| `README.md` | Add attention optimization section |

## Verification

1. Start server without sageattention → confirm SDPA flash works (baseline)
2. Install sageattention → confirm log says "SageAttention enabled"
3. Run Sniper preset → compare inference times vs baseline
4. Uninstall sageattention → confirm fallback works cleanly

## Documentation (do when implementing install guides)

- README quick start: mention SageAttention is auto-detected, no action needed for Docker
- README troubleshooting: "if inference seems slow, check server log for SageAttention enabled"
- Install guide (non-Docker): optional pip install command for power users
- Docker guide: nothing needed, it's baked in

## Test result: SageAttention v1 is SLOWER (2026-02-18)

Tested v1.0.6 (pip) with Sniper preset (512 image tokens, TTS enabled).
Result: ~50% slower than PyTorch SDPA flash baseline. Uninstalled.

Cause: PyTorch 2.7 SDPA flash kernels are native CUDA, well-optimized for Ada.
SageAttention v1 Triton JIT kernels add overhead that doesn't pay off at our
short sequence lengths (~700 tokens).

**v1 is ruled out.** Only v2 (INT4+FP8, compiled CUDA) is worth trying.
Test v2 in Docker build only — don't bother with local source compilation.
If v2 is also slower, drop SageAttention entirely and keep SDPA flash.

## Open questions

- Does SageAttention v2 perform better than v1 at short sequences (~700 tokens)?
  The compiled CUDA kernels (vs Triton JIT) may make the difference.
- Which branch/tag of the repo contains v2 code?
- If v2 is also slower: accept SDPA flash as the optimal backend and close this plan.
