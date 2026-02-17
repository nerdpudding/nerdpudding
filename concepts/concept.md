# Video Chat with AI - Concept Document

## Vision

Build a local, GPU-accelerated application that allows a user to **stream live video** (from any source) into a multimodal AI model and **have a conversation about what it sees** -- in real-time.

The system should feel like "chatting with someone who is watching the same video as you."

---

## Core Idea

```
+-------------------+       +-------------------+       +-------------------+
|   Video Source    | ----> |   AI Model        | ----> |   User Interface  |
|                   |       |   (MiniCPM-o 4.5) |       |                   |
| - Webcam          |       |                   |       | - Video display   |
| - VLC stream      |       | Processes video   |       | - Chat (text)     |
| - Phone camera    |       | frames + audio    |       | - Chat (voice)*   |
| - Video file      |       | in real-time      | <---- | - User input      |
+-------------------+       +-------------------+       +-------------------+
```

The key insight: from the model's perspective, **all video input looks the same** -- it's just frames. Whether those frames come from a physical webcam, a virtual camera device (v4l2loopback), or a video file being streamed doesn't matter. This gives us flexibility in input sources without changing the core application.

---

## System Context (C4 Level 1)

```
+-----------------------------------------------------------------------+
|                          User                                         |
|   Watches video, asks questions (text or voice),                      |
|   receives AI responses (text or speech)                              |
+----------------------------------+------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                    Video Chat Application                             |
|                                                                       |
|   Processes live video streams with MiniCPM-o 4.5 model              |
|   Enables real-time conversation about video content                  |
|   Runs locally on consumer GPU hardware                              |
+----------------------------------+------------------------------------+
                                   |
                      +------------+------------+
                      |                         |
                      v                         v
        +--------------------+     +------------------------+
        |   Video Sources    |     |   MiniCPM-o 4.5 Model |
        |                    |     |                        |
        | Webcam, VLC,       |     | HuggingFace / GGUF    |
        | phone camera,      |     | Running on local GPU   |
        | video files         |     |                        |
        +--------------------+     +------------------------+
```

---

## Container View (C4 Level 2)

```
+-----------------------------------------------------------------------+
|                       Video Chat Application                          |
|                                                                       |
|  +------------------+    +--------------------+    +----------------+ |
|  |  Video Input     |    |  Model Server      |    |  Web Frontend  | |
|  |  Layer           |--->|  (Python/FastAPI)   |--->|  (Browser UI)  | |
|  |                  |    |                     |    |                | |
|  | - Camera capture |    | - Model inference   |    | - Video view   | |
|  | - Virtual cam    |    | - Frame processing  |    | - Chat panel   | |
|  | - Stream ingest  |    | - WebSocket server  |    | - Text input   | |
|  |                  |    | - Audio processing* |    | - Voice input* | |
|  +------------------+    +--------------------+    +----------------+ |
|                                   |                                   |
|                                   v                                   |
|                          +----------------+                           |
|                          | Local GPU      |                           |
|                          | (RTX 4090)     |                           |
|                          +----------------+                           |
+-----------------------------------------------------------------------+

* = later iteration
```

---

## Input/Output Design

### Input Sources (flexible, iterative)

| Phase | Source | Method |
|-------|--------|--------|
| MVP | Local video file | v4l2loopback virtual camera or direct file feed |
| MVP | Webcam | Direct browser/WebRTC camera access |
| Later | Phone camera | Stream over network (IP camera / WebRTC) |
| Later | VLC / external stream | RTSP/RTMP to virtual camera device |

### User Interaction (iterative)

| Phase | Input | Output |
|-------|-------|--------|
| MVP | Text chat | Text response |
| Later | Voice input | Text response |
| Later | Voice input | Speech response (TTS) |

---

## Model Selection

**Target: MiniCPM-o 4.5** -- the latest omni-modal model supporting vision + audio + text.

| Variant | VRAM Required | Notes |
|---------|---------------|-------|
| Full (BF16) | ~19 GB | Fits on RTX 4090 (24 GB) |
| GGUF (quantized) | ~10 GB | Lower VRAM, llama.cpp backend |
| AWQ (quantized) | ~11 GB | Quantized weights, good perf |

**Primary target**: Full model on RTX 4090 (19 GB fits within 24 GB).
**Fallback**: GGUF or AWQ quantized if VRAM becomes an issue during streaming.

Sources:
- HuggingFace: `openbmb/MiniCPM-o-4_5-gguf`
- ModelScope: `OpenBMB/MiniCPM-o-4_5`

---

## Available Hardware

| Component | Spec | Role |
|-----------|------|------|
| **GPU (primary)** | NVIDIA RTX 4090 24 GB | Model inference, primary compute |
| **GPU (secondary)** | NVIDIA RTX 5070 Ti 16 GB (~12 GB usable) | Backup/overflow only, not primary target |
| **CPU** | AMD Ryzen 5800X3D | General compute, preprocessing |
| **RAM** | 64 GB DDR4 | Model loading, frame buffering |
| **OS** | Ubuntu Desktop | Development and runtime |

**VRAM constraints**: The 4090 with 24 GB is the main workhorse. The full MiniCPM-o 4.5 model at ~19 GB leaves ~5 GB headroom for inference context. This is tight but workable for streaming. If VRAM becomes a bottleneck (long context, high-res frames), we fall back to quantized variants.

**Mixed GPU note**: Using both GPUs adds complexity (different architectures, offloading logic). This is a last resort, not the default plan.

---

## Available Resources (Cloned Repos)

Two repositories are cloned inside this project workspace that contain implementations and demos we can build upon:

### 1. MiniCPM-o (`./MiniCPM-o/`)
The official model repository. Contains:
- **Web demos** (Gradio, Streamlit) for basic vision/chat interaction
- **FastAPI model server** with WebSocket streaming (`web_demos/minicpm-o_2.6/model_server.py`)
- **Vue 3 web frontend** with Dockerfile (`web_demos/minicpm-o_2.6/web_server/`)
- **VAD (Voice Activity Detection)** utilities for audio processing
- Model loading code, inference examples, quantization support

### 2. MiniCPM-V-CookBook (`./MiniCPM-V-CookBook/`)
A cookbook with production-ready demos. Contains:
- **WebRTC Demo** -- full-duplex real-time video interaction (most advanced, includes Docker)
- **Omni Stream Demo** -- Node.js/Vue voice+video streaming
- **Gradio Demo** -- simple web interface
- **OpenWebUI integration** -- Docker-based web UI
- **Inference examples** -- vision, audio, speech, video understanding
- Docker Compose files for multiple demo types

These repos provide existing implementations of much of what we need. The approach is to evaluate, adapt, and build upon them rather than starting from scratch.

---

## Use Cases

### Primary
1. **Live video conversation** -- Watch a video (or camera feed) and ask the AI questions about what's happening in real-time

### Secondary (future iterations)
2. **Surveillance/monitoring** -- AI describes events, triggers alerts on specific conditions
3. **Content logging** -- Automatic text logs of video content with timestamps
4. **Accessibility** -- Rich scene descriptions for visually impaired users
5. **Content creation pipeline** -- Generate detailed scene descriptions for video generation prompts
6. **Multi-model pipeline** -- Vision output feeds into other LLMs, alert systems, or databases

### Pipeline Vision (future)

```
Camera/Video --> MiniCPM-o (vision) --> Text descriptions -->
  |-- LLM (analysis, decisions)
  |-- Alert system (pattern matching, triggers)
  |-- Log/DB (searchable video archive)
  |-- Video gen model (scene recreation prompts)
```

---

## Development Approach

- **Proof of Concept** -- not enterprise-level
- **Iterative sprints** -- start minimal, find limitations, improve
- **SOLID, DRY, KISS** -- clean architecture but no over-engineering
- **Flexible** -- easy to swap video sources, model variants, or UI components
- **Local-first** -- everything runs on the local machine, no cloud dependencies
- **Docker where possible** -- containerized services for reproducibility
