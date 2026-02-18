# Video Chat with AI

Local, GPU-accelerated application that streams live video into a multimodal AI model for real-time commentary with text-to-speech. Point it at a football match, a security camera, a nature stream, or any video source — and get a live AI commentator that sees, understands, and speaks about what's happening.

**Status:** Sprint 2 complete. Core pipeline works end-to-end: video in, text + TTS audio out, with adaptive pacing. Docker and WebRTC are next (Sprint 3). See [Roadmap](#current-status) for details.

## Table of Contents

- [Demo: Live Sports Commentary](#demo-live-sports-commentary)
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

## Demo: Live Sports Commentary

The most fun way to try this: download a football match (or any sports broadcast) and let the AI commentate live — with voice.

```bash
# Start with TTS enabled
ENABLE_TTS=true python -m app.main
```

Open `http://localhost:8199` in your browser, enter the path to a video file, click **Start**, and set the instruction:

```
Commentate on this football match between Brazil (BRA) and France (FRA).
The scoreboard shows country abbreviations, the score, and the match clock
— the clock is NOT the score. Focus on exciting moments: attacks, shots,
saves, fouls, corners, and near-misses. Build tension during dangerous plays.
Be enthusiastic about goal chances, not monotone. Skip boring buildup in
midfield — only speak when something interesting happens.
```

Adapt the team names and context to your match. The AI will commentate with natural pacing — more during action, quieter during slow moments. Use the speaker button in the header to mute/unmute.

**Tip:** The prompt makes a big difference. Experiment with it while the video is running — you can change the instruction at any time. For example, the model may read the match clock too often. Adding a constraint like *"You may mention the match time only at 5, 10, 15, ... 90 minutes play time"* fixes that. See the [Tuning Guide](docs/tuning_guide.md#prompt-tips) for more prompt examples.

## Video Sources

Enter any of these in the "Video source" field in the browser UI:

| Source | Format | Example |
|--------|--------|---------|
| Local video file | File path | `/home/user/match.mp4` |
| Webcam | Device ID (integer) | `0` |
| RTSP stream | RTSP URL | `rtsp://192.168.1.100:554/stream` |
| HTTP MJPEG stream | HTTP URL | `http://192.168.1.100:8080/video` |
| HTTP video stream | HTTP URL | `http://example.com/stream.mp4` |

The system uses OpenCV's `VideoCapture` underneath, so anything OpenCV supports will work. Video files loop automatically for testing.

**Phone as camera:** Install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) (Android) or similar app, then use the MJPEG URL it provides (e.g. `http://192.168.1.50:8080/video`).

**VLC re-streaming:** Stream any content as RTSP from another PC:
```bash
vlc input.mp4 --sout '#rtp{sdp=rtsp://:8554/stream}'
# Then use: rtsp://<that-pc-ip>:8554/stream
```

**YouTube / Twitch:** Not supported directly. Use `yt-dlp -g <url>` to extract the direct stream URL, then paste that URL — but results vary depending on format and DRM.

## Getting Started

### Prerequisites

- NVIDIA GPU with sufficient VRAM (see table below)
- CUDA 12.x installed
- Miniconda or Anaconda
- ~10 GB disk space for AWQ model + TTS assets (~30 GB if also downloading BF16)

| Mode | VRAM Required | Tested On |
|------|--------------|-----------|
| Text-only (AWQ) | ~8.6 GB | RTX 4090 |
| Text + TTS (AWQ) | ~14-15 GB | RTX 4090 |
| Text-only (BF16) | ~18.5 GB | RTX 4090 |

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
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir models/MiniCPM-o-4_5 --include "assets/*"
cp -r models/MiniCPM-o-4_5/assets models/MiniCPM-o-4_5-awq/assets

# Optional: download full BF16 model (~19 GB, for comparison or fallback)
# huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir models/MiniCPM-o-4_5

# 4. Apply required model patches (see docs/model_patches.md for all patches)
#    AWQ model needs config.json fix + streaming fix in modeling_minicpmo.py
#    BF16 model (if downloaded) needs streaming fix in modeling_minicpmo.py

# 5. Start the server (text-only)
python -m app.main

# Or with TTS audio commentary
ENABLE_TTS=true python -m app.main

# Server starts on http://localhost:8199
```

Open the browser, enter a video source, click **Start**, type an instruction, and press **Send**. The AI commentary streams as text in the right panel. With TTS enabled, you'll also hear it — use the speaker button to mute/unmute.

### Testing Without a Browser

```bash
# Test model loading + inference on a single image
python -m scripts.test_model --image test_files/images/test.jpg

# Test frame capture from a video file
python -m scripts.test_capture --source test_files/videos/test.mp4

# Test full pipeline (model + capture + commentary loop)
python -m scripts.test_monitor --source test_files/videos/test.mp4 --cycles 2

# Test TTS audio output (saves WAV file)
ENABLE_TTS=true python -m scripts.test_tts --source test_files/videos/test.mp4
```

### Configuration

All settings are in `app/config.py` and overridable via environment variables. For detailed tuning instructions — including per-GPU recommendations, TTS pacing, scene detection, and prompt tips — see the **[Tuning Guide](docs/tuning_guide.md)**.

Quick examples:

```bash
# Enable TTS with custom pacing
ENABLE_TTS=true TTS_PAUSE_AFTER=1.5 python -m app.main

# Use BF16 model instead of AWQ (needs ~18.5 GB VRAM)
MODEL_PATH=models/MiniCPM-o-4_5 python -m app.main

# Disable video-commentary sync (show real-time video, no delay)
STREAM_DELAY_INIT=0 python -m app.main

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

**Sprint 2 complete.** Full end-to-end pipeline: video in, text + TTS audio out, with adaptive pacing and scene-weighted commentary density. See [Sprint 2 Review](docs/sprint2/SPRINT2_REVIEW.md) for detailed findings.

| Metric | Text-only | With TTS |
|--------|-----------|----------|
| VRAM (AWQ INT4) | ~8.6 GB | ~14-15 GB |
| Inference per cycle | ~1.6s avg | ~5s avg |
| End-to-end latency | ~4.8s avg | Audio-gated (adaptive) |
| Display frame rate | Native (~24 FPS via MJPEG) | Same |
| Commentary output | Streaming text (SSE) | Text + audio (Web Audio API) |

**Next:** Sprint 3 — Docker, LiveKit WebRTC, input robustness, UI polish.

## Documentation

- [Tuning Guide](docs/tuning_guide.md) -- per-GPU settings, TTS pacing, prompt tips
- [AI Instructions](AI_INSTRUCTIONS.md) -- project rules, hierarchy, agent table
- [Detailed Concept](concepts/concept.md) -- full concept with diagrams and technical details
- [Roadmap](roadmap.md) -- project roadmap and sprint status
- [Sprint 1 Review](docs/sprint1/SPRINT1_REVIEW.md) -- sprint summary, findings, Sprint 2 ideas
- [Sprint 1 Log](docs/sprint1/SPRINT1_LOG.md) -- setup steps, test results, findings
- [Sprint 2 Review](docs/sprint2/SPRINT2_REVIEW.md) -- sprint summary, findings, Sprint 3 recommendations
- [Sprint 2 Log](docs/sprint2/SPRINT2_LOG.md) -- AWQ, latency, MJPEG sync, TTS, audio pacing
- [Model Patches](docs/model_patches.md) -- patches applied to model files (must reapply after update)
- [Lessons Learned](docs/lessons_learned.md) -- what worked and didn't (context for AI assistants)
- [docs/](docs/) -- all guides and reference documentation
