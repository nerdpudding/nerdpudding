# Tasks - February 19, 2026

## Priority 1: Audio-commentary pacing (awaiting approval)

Plan: `claude_plans/PLAN_audio_pacing.md` — submitted for review.

- [ ] Step A: Audio-gated pacing (fixes drift)
- [ ] Step B: Inter-cycle breathing pause
- [ ] Step C: MAX_NEW_TOKENS cap for TTS mode
- [ ] Step D: Scene-weighted commentary length

## Sprint 2 remaining steps

- [ ] Step 5: Docker setup (Dockerfile, docker-compose, .env.example, LiveKit config)
- [ ] Step 6: LiveKit WebRTC (bot, frame_provider, token endpoint, frontend)
- [ ] Step 7: Input robustness (reconnect logic, status reporting, test matrix)
- [ ] Step 8: UI updates (source mode selector, MJPEG, LiveKit player, TTS controls)

## Completed (Feb 17-18)

- [x] Step 1: AWQ model support
- [x] Step 1b: Latency optimization
- [x] Step 2: MJPEG streaming with adaptive sync
- [x] Step 3: TTS integration
- [x] Step 4: Audio delivery pipeline + browser playback
