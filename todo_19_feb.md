# Tasks - February 19, 2026

## Carry-over: Audio timing optimization

Step 4 audio delivery works but pacing needs tuning. With TTS enabled, inference cycles fire faster than audio can naturally play back. Need to explore:
- Shorter responses via prompt tuning when TTS active
- Increased INFERENCE_INTERVAL for TTS mode
- Cooldown after audio finishes before next cycle
- Dynamic timing based on audio queue depth

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
