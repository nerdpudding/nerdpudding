# AI Instructions — Video Chat with AI

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
- **Build on existing work** — evaluate and adapt code from the cloned repos before writing from scratch.
- **Local-first** — everything runs on the local machine, no cloud dependencies.
- **Docker where possible** — containerized services for reproducibility.

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
├── README.md                         # Project overview, hardware, use cases
├── roadmap.md                        # Project roadmap and sprint status
├── todo_feb_17.md                    # Daily task tracker
├── concepts/
│   └── concept.md                    # Detailed concept: architecture, model selection, constraints
├── docs/                             # Guides, tutorials, and reference documentation
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

**Note:** The cloned repos (`MiniCPM-o/`, `MiniCPM-V-CookBook/`) are reference material. Read from them freely, but do not modify their contents. Our own application code will live in separate directories as it gets built.

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
2. Current task tracker if one exists (e.g. `todo_feb_17.md`)
3. Active plans in `claude_plans/`
4. `concepts/concept.md` for project context
5. Then continue with the task
