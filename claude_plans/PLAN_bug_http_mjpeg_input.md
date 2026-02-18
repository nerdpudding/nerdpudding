# BUG: HTTP MJPEG streams fail as video source

## Context

NerdPudding accepts video sources via the `/api/start` endpoint: device IDs, file paths, RTSP URLs, and HTTP URLs. All are passed directly to `cv2.VideoCapture(source)`. This works for most source types, but fails for HTTP MJPEG streams (`multipart/x-mixed-replace` format) — the most common format used by IP cameras, browser-based streams, and tools like the NerdCam webcam app.

OpenCV/FFmpeg cannot auto-detect the multipart boundary format. FFmpeg sees the boundary strings (`--ffmpeg\r\n`, `Content-Type: image/jpeg\r\n`, etc.) as unparseable data and reports `Stream ends prematurely at 0`.

VLC handles these streams fine because it has a dedicated multipart/x-mixed-replace parser. NerdPudding needs the same.

## Root cause

`frame_capture.py:73` passes all sources blindly to `cv2.VideoCapture()`:
```python
self._capture = cv2.VideoCapture(source)
```

No detection of source type, no fallback for HTTP MJPEG.

## Fix

Add HTTP MJPEG support to `FrameCapture` by detecting HTTP URLs and probing their Content-Type before choosing a read strategy.

### Changes to `app/frame_capture.py`

1. **Add source type detection** in `start()`:
   - If source is an `int` → device ID, use `cv2.VideoCapture` (existing path)
   - If source is a `str` starting with `http://` or `https://` → probe Content-Type with a HEAD request
     - If `multipart/x-mixed-replace` → use custom MJPEG-over-HTTP reader
     - Otherwise → use `cv2.VideoCapture` (existing path, handles HTTP video files)
   - If source is any other `str` → file path or RTSP URL, use `cv2.VideoCapture` (existing path)

2. **Add `_mjpeg_http_loop()` method** — a new capture loop for HTTP MJPEG streams:
   - Opens HTTP GET connection using `urllib.request.urlopen` (stdlib, no new deps)
   - Reads the response as a byte stream
   - Parses multipart boundaries (extracts `Content-Length` from part headers, reads that many JPEG bytes)
   - Falls back to JPEG marker scanning (`FFD8`/`FFD9`) if Content-Length is missing
   - Decodes each JPEG into a PIL Image
   - Feeds into the same two output paths: display buffer + inference callback (identical to the existing `_capture_loop`)

3. **Refactor shared logic**: The display-buffer push and inference-callback push are identical between `_capture_loop` and `_mjpeg_http_loop`. Extract into a small `_process_frame(pil_image)` method to avoid duplication.

### What stays the same

- `app/main.py` — no changes, `StartRequest` already accepts `str | int`
- `app/config.py` — no new config vars needed
- `app/static/index.html` — no changes, source input already accepts URLs
- `app/sliding_window.py` — untouched
- `app/monitor_loop.py` — untouched

### No new dependencies

Uses only `urllib.request` (stdlib). No `requests`, no `httpx`.

## Files to modify

| File | Change |
|------|--------|
| `app/frame_capture.py` | Source detection, MJPEG HTTP reader, refactor shared frame processing |

## Verification

1. **HTTP MJPEG stream** (NerdCam webcam app):
   - Start webcam app server on port 8088
   - Start NerdPudding, set source to `http://localhost:8088/api/mjpeg`
   - Verify frames appear in browser MJPEG view and inference runs

2. **Video file** (existing path, must not break):
   - Start NerdPudding with source `test_files/videos/test.mp4`
   - Verify works as before

3. **Device ID** (existing path, must not break):
   - Start NerdPudding with source `0`
   - Verify webcam capture works as before

4. **RTSP stream** (if available):
   - Test with an RTSP URL directly
   - Verify cv2.VideoCapture path still works

5. **HTTP non-MJPEG** (edge case):
   - If a direct HTTP link to an mp4 file is available, verify it still works via cv2.VideoCapture
