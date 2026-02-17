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

```bash
git clone <your-repo-url>
cd video_chat

# Clone the reference repos (not included in this repo)
git clone https://github.com/OpenBMB/MiniCPM-o.git
git clone https://github.com/OpenBMB/MiniCPM-V-CookBook.git
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

## Documentation

- [AI Instructions](AI_INSTRUCTIONS.md) -- project rules, hierarchy, agent table
- [Detailed Concept](concepts/concept.md) -- full concept with diagrams and technical details
- [Roadmap](roadmap.md) -- project roadmap
- [Sprint 1 Log](SPRINT1_LOG.md) -- setup steps, test results, findings
- [Model Patches](docs/model_patches.md) -- patches applied to model files (must reapply after update)
- [Lessons Learned](docs/lessons_learned.md) -- what worked and didn't (context for AI assistants)
- [docs/](docs/) -- all guides and reference documentation
