# AI Instructions — NerdPudding

## Project overview

Proof-of-concept application for real-time video conversation with a multimodal AI model (MiniCPM-o 4.5). Streams live video from flexible sources (webcam, VLC, video files via virtual camera) into a locally hosted model on consumer GPU hardware, with a web-based chat interface. Built iteratively — start minimal, find limitations, improve.

## Principles

- **SOLID, DRY, KISS** — always. No over-engineering, no premature abstractions.
- **One source of truth** — no duplicate information across files.
- **Never delete, always archive** — outdated content goes to `archive/`.
- **Modularity** — keep video input, model server, and frontend cleanly separated.
- **Keep everything up to date** — after any change, verify that READMEs, docs, agent instructions, and config files still reflect reality. Stale docs are worse than no docs.
- **Use agents when their role fits** — don't do manually what an agent is designed for. Check the agents table below before starting a task.
- **ALL code, docs, comments, plans, and commit messages MUST be in English** — always, no exceptions. The user often communicates in Dutch, but everything written to files must be English.
- **Learn from mistakes** — when an approach fails or wastes effort, document it in `docs/lessons_learned.md`. This file is persistent context for AI assistants to avoid repeating the same mistakes. Update it whenever a significant lesson emerges (failed approach, wrong assumption, better workflow discovered).
- **Research before coding** — when integrating with external code (model libraries, APIs, demos), thoroughly trace the actual code paths before writing integration code. Don't iterate by trial-and-error. Use the repo-researcher agent for this.
- **Build on existing work** — evaluate and adapt code from the cloned repos before writing from scratch.
- **Local-first** — everything runs on the local machine, no cloud dependencies.
- **Docker where possible** — containerized services for reproducibility.
- **Security-conscious** — follow the security guidelines below, even for research code.

## Security

This is research/experimental software, not a production service. But since the repo is public, follow these practices:

### Current safeguards
- Server binds to `127.0.0.1` by default (localhost only). Override with `SERVER_HOST=0.0.0.0` for Docker or network access.
- No authentication — acceptable for local use, must be added before exposing on untrusted networks.
- Subscriber queues are bounded (drop data for slow consumers, prevent memory exhaustion).
- Instruction input has a length limit (2000 chars).
- `trust_remote_code=True` is required for MiniCPM-o — model files are user-downloaded, not fetched from untrusted sources.

### Rules for new code
- **Never use `eval()`, `exec()`, `os.system()`, or `subprocess` with `shell=True`**.
- **Never pass user input to file path operations** without validation. If accepting paths from users (API endpoints), validate they don't contain `..` or traverse outside expected directories.
- **Bound all queues and buffers** — use `maxsize` on `asyncio.Queue`, `maxlen` on `deque`. Never allow unbounded growth from external input.
- **Don't leak internal errors to clients** — log the full error, return a generic message.
- **Keep secrets out of code** — use environment variables. `.env` files are gitignored.
- **Pin dependency versions** — specify exact versions in requirements.txt.

### Docker security (Sprint 3)
When containerizing:
- Run as non-root user (`USER appuser` in Dockerfile)
- Don't expose ports without documenting the security implications
- Mount model files read-only (`:ro`)
- Use multi-stage builds to minimize image attack surface
- Never bake secrets or model files into the image

## Workflow

For non-trivial changes, follow this order:

1. **Plan** — create a plan in `claude_plans/` with a logical name
2. **Ask for approval** — present the plan to the user before implementing
3. **Implement** — follow the approved plan, use the best approach
4. **Test** — verify changes work (sometimes manual with user involvement)
5. **Iterate** — if tests reveal issues, fix and re-test
6. **Clean up** — archive completed plans, remove unused files (to archive), update docs and agent instructions if affected

## Project hierarchy

```
.
├── AI_INSTRUCTIONS.md                # THIS FILE — read first
├── README.md                         # Project overview, hardware, getting started guide
├── roadmap.md                        # Project roadmap and sprint status
├── todo_<date>.md                    # Daily task tracker (when active)
├── app/                              # Application code
│   ├── __init__.py
│   ├── config.py                     # All configuration (env var overridable, single source of truth)
│   ├── model_server.py               # Model loading + streaming inference
│   ├── frame_capture.py              # Background thread capture (OpenCV)
│   ├── sliding_window.py             # Thread-safe ring buffer with FrameMeta
│   ├── monitor_loop.py               # Async orchestrator: IDLE/ACTIVE modes, pub/sub output
│   ├── audio_manager.py              # TTS audio resampling (24kHz→48kHz) + pub/sub delivery
│   ├── main.py                       # FastAPI server (REST + SSE + audio stream endpoints)
│   ├── static/
│   │   └── index.html                # Web UI (vanilla HTML/JS/CSS)
│   └── requirements.txt              # Python dependencies
├── scripts/                          # Standalone utility scripts
│   ├── test_model.py                 # Model loading + inference test
│   ├── test_capture.py               # Frame capture + sliding window test
│   ├── test_monitor.py               # End-to-end monitor loop test
│   └── test_tts.py                   # TTS quality/latency test script
├── models/                           # Downloaded model files (git-ignored)
│   ├── MiniCPM-o-4_5/               # Full BF16 model + patched model code (~19 GB)
│   └── MiniCPM-o-4_5-awq/           # AWQ INT4 model + patched config + code (~8 GB)
├── test_files/                       # Test assets (images, videos, audio output)
│   ├── images/
│   ├── videos/
│   └── audio/                        # TTS test outputs (WAV files)
├── concepts/
│   └── concept.md                    # Detailed concept: architecture, model selection, constraints
├── docs/                             # Guides, tutorials, and reference documentation
│   ├── tuning_guide.md               # Per-GPU settings, presets, TTS pacing, prompt tips
│   ├── tuning_test_results.md        # Preset benchmarks, attention backend comparisons
│   ├── model_patches.md              # Patches applied to model files (must reapply after update)
│   ├── lessons_learned.md            # What worked and didn't (context for AI assistants)
│   ├── sprint1/                      # Sprint 1 deliverables
│   │   ├── SPRINT1_REVIEW.md         # Sprint summary, findings, Sprint 2 recommendations
│   │   └── SPRINT1_LOG.md            # Step-by-step progress log with test results
│   └── sprint2/                      # Sprint 2 deliverables
│       ├── SPRINT2_REVIEW.md         # Sprint summary, findings, Sprint 3 recommendations
│       └── SPRINT2_LOG.md            # Step-by-step progress log
├── MiniCPM-o/                        # Official model repo (cloned, do not modify)
├── MiniCPM-V-CookBook/              # Cookbook with demos (cloned, do not modify)
├── claude_plans/                     # Active plans (see Plan rules below)
├── archive/                          # Archived plans, old docs, superseded files
└── .claude/
    └── agents/                       # Claude Code specialized agents
        ├── doc-keeper.md             # Documentation audit and maintenance
        ├── builder.md                # Build, Docker, service orchestration, code adaptation
        ├── environment-setup.md      # Python envs, dependencies, model downloads, Docker config
        └── repo-researcher.md        # Read-only research of cloned reference repos
```

**Note:** The cloned repos (`MiniCPM-o/`, `MiniCPM-V-CookBook/`) are reference material. Read from them freely, but do not modify their contents. The `models/` directory contains downloaded model files (git-ignored) with a required patch -- see `docs/model_patches.md`.

## Agents

Use agents when their role matches the task. Don't reinvent what an agent already handles. Agent files live in `.claude/agents/`.

| Agent | When to use |
|-------|-------------|
| `doc-keeper` | Documentation audit, consistency checks, hierarchy maintenance, cross-reference updates, archiving recommendations |
| `builder` | Dockerfiles, docker-compose, build scripts, startup scripts, wiring components together, adapting code from cloned repos |
| `environment-setup` | Python environments, dependency installation, model downloads, CUDA/GPU config, Docker container troubleshooting |
| `repo-researcher` | Read-only exploration of `MiniCPM-o/` and `MiniCPM-V-CookBook/` -- demo comparison, code tracing, dependency extraction |

After changes that affect an agent's domain, update that agent's instructions.

## Plan rules

Plans are stored in: **`claude_plans/`**

1. **Always save plans as files** — plans must be persistent, never just in conversation.
2. **Use logical names** — e.g. `PLAN_setup_model_server.md`. If plan mode generates a random name, rename it immediately.
3. **No duplicates** — if a plan already exists for the same topic, update it instead of creating a new one.
4. **Archive when done** — completed plans move to `archive/` with a date prefix: `2026-02-17_setup_model_server.md`.

## Archive rules

Everything goes to: **`archive/`**

- Completed plans (from `claude_plans/`)
- Superseded documentation
- Old scripts replaced by new ones
- Outdated daily schedules or todo files

Never delete files. Always archive.

## Git commits

- Write normal, descriptive commit messages.
- Never add "Co-Authored-By: Claude" or AI attribution.
- Only commit when explicitly asked.

## After compaction

When resuming after compaction, read in this order:
1. This file (`AI_INSTRUCTIONS.md`)
2. Current task tracker if one exists (check root for a `todo_*.md` file)
3. Active plans in `claude_plans/`
4. `concepts/concept.md` for project context
5. Then continue with the task
