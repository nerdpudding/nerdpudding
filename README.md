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

- NVIDIA GPU with 20+ GB VRAM (tested on RTX 4090, 16.4 GB used)
- CUDA 12.x installed
- Miniconda or Anaconda
- ~20 GB disk space for model files

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

# 2. Download model (~19 GB)
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir models/MiniCPM-o-4_5

# 3. Apply required model patch (see docs/model_patches.md for details)
#    One-line fix in models/MiniCPM-o-4_5/modeling_minicpmo.py

# 4. Start the server
python -m app.main
# Server starts on http://localhost:8199

# 5. Open browser to http://localhost:8199
#    - Enter a video source (file path, device ID, or stream URL)
#    - Click Start, type an instruction, watch the AI commentary stream in
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
# Lower latency: fewer frames, higher capture rate
FRAMES_PER_INFERENCE=4 CAPTURE_FPS=2 python -m app.main

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
| Full (BF16) | ~19 GB | Python / transformers | [HuggingFace](https://huggingface.co/openbmb/MiniCPM-o-4_5) / [ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5) |
| GGUF (quantized) | 4.8 - 16.4 GB | C++ / llama.cpp | [HuggingFace](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf) |

**Primary target:** Full BF16 on RTX 4090. **Fallback:** GGUF if VRAM constrained. See [concept](concepts/concept.md#model-selection) for detailed comparison.

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

**Sprint 1 complete. Sprint 2 in progress.** Working MVP from Sprint 1: continuous video monitoring with real-time AI commentary, steerable mid-stream. Sprint 2 adds AWQ INT4 model, TTS audio output, LiveKit WebRTC, and Docker. See [Sprint 2 Plan](claude_plans/PLAN_sprint2.md) for details.

| Metric | Value |
|--------|-------|
| Model VRAM usage | 16.4 GB |
| Inference per cycle | 1.2-2.3s |
| End-to-end latency | 8-11s |
| Capture rate | 1 FPS |

## Documentation

- [AI Instructions](AI_INSTRUCTIONS.md) -- project rules, hierarchy, agent table
- [Detailed Concept](concepts/concept.md) -- full concept with diagrams and technical details
- [Roadmap](roadmap.md) -- project roadmap and sprint status
- [Sprint 1 Review](docs/sprint1/SPRINT1_REVIEW.md) -- sprint summary, findings, Sprint 2 ideas
- [Sprint 1 Log](docs/sprint1/SPRINT1_LOG.md) -- setup steps, test results, findings
- [Model Patches](docs/model_patches.md) -- patches applied to model files (must reapply after update)
- [Lessons Learned](docs/lessons_learned.md) -- what worked and didn't (context for AI assistants)
- [docs/](docs/) -- all guides and reference documentation
