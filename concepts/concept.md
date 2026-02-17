# Video Chat with AI - Concept Document

## Vision

Build a local, GPU-accelerated application that allows a user to **stream live video** (from any source) into a multimodal AI model and **have a conversation about what it sees** -- in real-time.

The system should feel like "chatting with someone who is watching the same video as you."

### What this means concretely

This is **not** a video upload tool. The user does not upload a video file and wait for the model to process it. Instead:

1. A **continuous video stream** plays (from any source -- webcam, VLC, OBS, video file via stream)
2. The model **continuously watches** the stream via a sliding window of recent frames
3. The user gives an instruction like "tell me what's happening" and the model starts **narrating live** as the stream progresses
4. The user can **steer the AI mid-stream** -- e.g., "now only tell me what the dog is doing when it's on screen" -- and the model adjusts its focus going forward
5. The model does **not** need to remember what happened an hour ago. A sliding window of recent context (last N seconds/frames) is sufficient.

Think of it as a **live commentator** that watches the same stream as you and responds to your directions in real-time.

### What this is NOT

- Not a video file upload + batch processing tool
- Not limited to short clips
- Not a single frame Q&A (though that's a building block)
- Not dependent on processing the entire video before responding

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

### Interaction Pattern

This is a **continuous monitoring loop**, not request-response:

```
Video Stream --> [Frame Buffer / Sliding Window] --> Model (periodic inference) --> Streaming text output
                                                          ^
                                                          |
                                              User steers via chat:
                                              "tell me what's happening"
                                              "focus on the dog only"
                                              "what just changed?"
```

The model continuously receives recent frames and generates output. The user can steer what the model focuses on at any time. Previous instructions remain active until replaced -- like directing a commentator.

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
|  | - Stream ingest  |    | - SSE streaming     |    | - Text input   | |
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

The streaming app (VLC, OBS, or similar) is a core part of the project -- researching the best option for providing a video stream to our system is part of Sprint 1.

| Phase | Source | Method |
|-------|--------|--------|
| MVP | Webcam | Direct camera access or via streaming app |
| MVP | Video file as stream | VLC, OBS, or similar app streaming to v4l2loopback or direct feed |
| MVP | Streaming app research | Evaluate VLC, OBS, FFmpeg, etc. as stream providers |
| Later | Phone camera | Stream over network (IP camera / WebRTC) |
| Later | External streams | RTSP/RTMP from other sources |

### User Interaction (iterative)

| Phase | Input | Output |
|-------|-------|--------|
| MVP | Text chat | Text response |
| Later | Voice input | Text response |
| Later | Voice input | Speech response (TTS) |

---

## Model Selection

**Target: MiniCPM-o 4.5** -- the latest omni-modal model supporting vision + audio + text in a single 9B parameter model.

### Why this model

MiniCPM-o 4.5 is chosen because it has everything we need built into one model:
- **Vision**: video understanding up to 10 FPS, any aspect ratio
- **STT/ASR**: speech-to-text built in (Whisper-based, multilingual)
- **TTS**: text-to-speech built in (with voice cloning)
- **Full-duplex mode**: simultaneous video+audio input with text+speech output (`model.as_duplex()`)
- Runs on consumer hardware (RTX 4090 with 24GB)

For Sprint 1 we use vision + text only. STT and TTS are already in the model and can be enabled later without switching models.

### Model variants

| Variant | Source | Inference backend | VRAM | Download |
|---------|--------|-------------------|------|----------|
| **AWQ INT4** (default) | [HuggingFace](https://huggingface.co/openbmb/MiniCPM-o-4_5-awq) | Python / transformers + autoawq | ~8.6 GB | ~8 GB (2 safetensor files) |
| **Full (BF16)** | [HuggingFace](https://huggingface.co/openbmb/MiniCPM-o-4_5) | Python / transformers | ~18.5 GB | ~18.7 GB (4 safetensor files) |
| **Full (BF16)** | [ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5) | Python / transformers | ~18.5 GB | Same model, Chinese mirror |
| **GGUF (quantized)** | [HuggingFace](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf) | C++ / llama.cpp | 4.8 - 16.4 GB | Various quantization levels |

### Key differences between variants

| | Full (BF16) | GGUF |
|---|---|---|
| **Inference** | Python + transformers + CUDA | C++ / llama.cpp (must compile) |
| **Quality** | Full precision, no loss | Quantized, slight quality loss depending on level |
| **VRAM** | ~19 GB (vision-only: less) | Q8_0: 8.7 GB, Q6_K: 6.7 GB, Q4_K_M: 5.0 GB |
| **Speed** | Model card claims 154 tokens/s (BF16), hardware unknown -- must benchmark on our 4090 | Model card claims 212 tokens/s (INT4), hardware unknown |
| **Audio/TTS** | Fully supported | Text-only in llama.cpp (no audio/TTS) |
| **Complexity** | pip install, done | Compile llama.cpp, different inference API |
| **Our use** | Sprint 1 primary; Sprint 2 fallback | Fallback if VRAM constrained |

### Decision

**Sprint 2+: AWQ INT4 from HuggingFace** (`openbmb/MiniCPM-o-4_5-awq`).
- 54% less VRAM (~8.6 GB vs ~18.5 GB), comparable output quality
- Same `AutoModel.from_pretrained()` loading — auto-detected via `quantization_config`
- Needs `autoawq` from custom fork (see `app/requirements.txt`)
- Leaves ~15 GB headroom on RTX 4090 for TTS, inference context, etc.
- BF16 remains available via `MODEL_PATH=models/MiniCPM-o-4_5` env var
- GGUF deferred to Sprint 3 (no TTS support in llama.cpp)

Previously (Sprint 1): Full BF16 was the default, which worked but left only ~5 GB headroom.

---

## Available Hardware

| Component | Spec | Role |
|-----------|------|------|
| **GPU (primary)** | NVIDIA RTX 4090 24 GB | Model inference, primary compute |
| **GPU (secondary)** | NVIDIA RTX 5070 Ti 16 GB (~12 GB usable) | Backup/overflow only, not primary target |
| **CPU** | AMD Ryzen 5800X3D | General compute, preprocessing |
| **RAM** | 64 GB DDR4 | Model loading, frame buffering |
| **OS** | Ubuntu Desktop | Development and runtime |

**VRAM constraints**: The 4090 with 24 GB is the main workhorse. With the default AWQ INT4 model at ~8.6 GB, there is ~15 GB headroom for TTS (~2 GB), inference context, and future features. The BF16 model at ~18.5 GB leaves only ~5 GB headroom — still workable but tight.

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
