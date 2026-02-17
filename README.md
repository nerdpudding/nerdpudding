# Video Chat with AI

Local, GPU-accelerated application for real-time video conversation with a multimodal AI model.

## Table of Contents

- [Getting Started](#getting-started)
- [Goal](#goal)
- [Architecture Overview](#architecture-overview)
- [Use Cases](#use-cases)
- [Model](#model)
- [Resources](#resources)
- [Hardware](#hardware)
- [Development Approach](#development-approach)
- [Project Structure & Agents](#project-structure--agents)
- [Documentation](#documentation)

## Getting Started

### Prerequisites

- NVIDIA GPU with ~10 GB VRAM (tested on RTX 4090, ~8.6 GB used with default AWQ model; BF16 fallback needs ~18.5 GB)
- CUDA 12.x installed
- Miniconda or Anaconda
- ~10 GB disk space for AWQ model + TTS assets (~30 GB if also downloading BF16)

### Quick Start

```bash
git clone <your-repo-url>
cd video_chat

# Clone the reference repos (not included in this repo)
git clone https://github.com/OpenBMB/MiniCPM-o.git
git clone https://github.com/OpenBMB/MiniCPM-V-CookBook.git

# 1. Create conda environment
conda create -n video_chat python=3.12 -y
conda activate video_chat
pip install -r app/requirements.txt

# 2. Download AWQ INT4 model (~8 GB, default)
huggingface-cli download openbmb/MiniCPM-o-4_5-awq --local-dir models/MiniCPM-o-4_5-awq

# 3. Download TTS assets (~1.2 GB vocoder + reference audio)
#    The AWQ model needs these from the BF16 model's assets directory.
#    Download BF16 model and copy assets, or download assets only:
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir models/MiniCPM-o-4_5 --include "assets/*"
cp -r models/MiniCPM-o-4_5/assets models/MiniCPM-o-4_5-awq/assets

# Optional: download full BF16 model (~19 GB, for comparison or fallback)
# huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir models/MiniCPM-o-4_5

# 4. Apply required model patches (see docs/model_patches.md for all patches)
#    AWQ model needs config.json fix + streaming fix in modeling_minicpmo.py
#    BF16 model (if downloaded) needs streaming fix in modeling_minicpmo.py

# 5. Start the server
python -m app.main
# Server starts on http://localhost:8199

# 6. Open browser to http://localhost:8199
#    - Enter a video source (file path, device ID, or stream URL)
#    - Click Start — smooth video plays at native frame rate (MJPEG)
#    - Type an instruction (e.g. "describe what you see") and press Send
#    - AI commentary streams in the right panel, synced with the video
#    - Video is shown with a ~5s adaptive delay that matches commentary timing
```

### Testing Without a Browser

```bash
# Test model loading + inference on a single image
python -m scripts.test_model --image test_files/images/test.jpg

# Test frame capture from a video file
python -m scripts.test_capture --source test_files/videos/test.mp4

# Test full pipeline (model + capture + commentary loop)
python -m scripts.test_monitor --source test_files/videos/test.mp4 --cycles 2
```

### Configuration

All settings are in `app/config.py` and overridable via environment variables:

```bash
# Use BF16 model instead of AWQ (needs ~18.5 GB VRAM)
MODEL_PATH=models/MiniCPM-o-4_5 python -m app.main

# Disable video-commentary sync (show real-time video, no delay)
STREAM_DELAY_INIT=0 python -m app.main

# Shorter responses
MAX_NEW_TOKENS=256 python -m app.main

# Different GPU
CUDA_VISIBLE_DEVICES=1 python -m app.main
```

These repos are used as reference material only -- see [Resources](#resources) for details.

## Goal

Stream live video from any source into **MiniCPM-o 4.5** and have a real-time conversation about what it sees -- like a live commentator that watches along and responds to your directions.

The AI **continuously monitors** the video stream and narrates or answers based on a sliding window of recent frames. The user can steer the AI's focus at any time (e.g., "only tell me what the dog does"). This is not video upload + batch processing -- it's live, continuous, and steerable. Text chat first, voice interaction later.

## Architecture Overview

```
Video Source  --->  Model Server (MiniCPM-o 4.5)  --->  Web UI
(cam/stream/file)       (Python, local GPU)           (browser)
                              ^                           |
                              |     user questions        |
                              +---------------------------+
```

## Use Cases

- **Live video conversation** -- ask questions about what the AI sees in real-time
- **Monitoring & alerting** -- describe events, trigger alerts on conditions
- **Content logging** -- auto-generate text summaries of video content
- **Accessibility** -- rich scene descriptions for visually impaired users
- **Multi-model pipeline** -- feed vision output into other LLMs, alert systems, or video generators

## Model

**[MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5)** -- omni-modal model (vision + audio/STT + TTS), 9B parameters. Supports video understanding up to 10 FPS, speech recognition, text-to-speech, and full-duplex streaming -- all in one model.

| Variant | VRAM | Backend | Link |
|---------|------|---------|------|
| AWQ INT4 (default) | ~8.6 GB | Python / transformers + autoawq | [HuggingFace](https://huggingface.co/openbmb/MiniCPM-o-4_5-awq) |
| Full (BF16) | ~18.5 GB | Python / transformers | [HuggingFace](https://huggingface.co/openbmb/MiniCPM-o-4_5) / [ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5) |
| GGUF (quantized) | 4.8 - 16.4 GB | C++ / llama.cpp | [HuggingFace](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf) |

**Primary target:** AWQ INT4 on RTX 4090 (~8.6 GB VRAM, comparable quality to BF16). **Fallback:** BF16 via `MODEL_PATH` env var. See [concept](concepts/concept.md#model-selection) for detailed comparison.

## Resources

Built upon two cloned repositories:

| Repo | Contents |
|------|----------|
| `MiniCPM-o/` | Official model repo -- web demos, FastAPI server, Vue frontend, VAD |
| `MiniCPM-V-CookBook/` | Cookbook -- WebRTC demo, Omni Stream, Gradio, Docker setups, inference examples |

## Hardware

| Component | Spec |
|-----------|------|
| GPU (primary) | NVIDIA RTX 4090 24 GB |
| GPU (secondary) | NVIDIA RTX 5070 Ti 16 GB (~12 GB usable) -- backup only |
| CPU | AMD Ryzen 5800X3D |
| RAM | 64 GB DDR4 |
| OS | Ubuntu Desktop |
| Tools | Docker, npm, miniconda, uv |

The RTX 4090 is the primary compute target. The 5070 Ti is available but only considered if VRAM constraints require multi-GPU offloading (adds complexity due to mixed architectures).

## Development Approach

Proof of concept with iterative sprints. Start minimal, find limitations, improve. SOLID, DRY, KISS.

## Project Structure & Agents

See the [Project hierarchy](AI_INSTRUCTIONS.md#project-hierarchy) in AI_INSTRUCTIONS.md for what each folder and file is for, including the agent table and usage guidelines.

## Current Status

**PoC milestone reached (Sprint 2, Step 2).** Smooth native-rate video with real-time AI commentary, adaptively synced, on consumer GPU hardware. Sprint 2 continues with TTS, LiveKit, and Docker (Steps 3-8). See [Sprint 2 Plan](claude_plans/PLAN_sprint2.md) for details. Git tag: `poc-milestone`.

| Metric | Value |
|--------|-------|
| Model VRAM (AWQ INT4) | ~8.6 GB |
| Inference per cycle | ~1.6s avg |
| End-to-end latency | ~4.8s avg (adaptively synced) |
| Display frame rate | Native (~24 FPS via MJPEG) |
| Inference capture rate | 2 FPS |

## Documentation

- [AI Instructions](AI_INSTRUCTIONS.md) -- project rules, hierarchy, agent table
- [Detailed Concept](concepts/concept.md) -- full concept with diagrams and technical details
- [Roadmap](roadmap.md) -- project roadmap and sprint status
- [Sprint 1 Review](docs/sprint1/SPRINT1_REVIEW.md) -- sprint summary, findings, Sprint 2 ideas
- [Sprint 1 Log](docs/sprint1/SPRINT1_LOG.md) -- setup steps, test results, findings
- [Sprint 2 Log](docs/sprint2/SPRINT2_LOG.md) -- AWQ, latency optimization, MJPEG adaptive sync
- [Model Patches](docs/model_patches.md) -- patches applied to model files (must reapply after update)
- [Lessons Learned](docs/lessons_learned.md) -- what worked and didn't (context for AI assistants)
- [docs/](docs/) -- all guides and reference documentation
