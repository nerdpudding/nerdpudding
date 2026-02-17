import asyncio
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    FRAME_JPEG_QUALITY,
    MJPEG_FPS,
    SERVER_HOST,
    SERVER_PORT,
    STREAM_DELAY_INIT,
)
from app.frame_capture import FrameCapture
from app.model_server import ModelServer
from app.monitor_loop import MonitorLoop
from app.sliding_window import SlidingWindow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


# --- Request/Response models ---


class StartRequest(BaseModel):
    source: str | int = 0


class InstructionRequest(BaseModel):
    instruction: str


class StatusResponse(BaseModel):
    model_loaded: bool
    capture_running: bool
    monitor_mode: str
    instruction: Optional[str]
    cycle_count: int
    frames_buffered: int


# --- App lifecycle ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    model = ModelServer()
    window = SlidingWindow()
    capture = FrameCapture(on_frame=window.push)
    monitor = MonitorLoop(model, window)

    app.state.model = model
    app.state.window = window
    app.state.capture = capture
    app.state.monitor = monitor
    app.state.monitor_task = asyncio.create_task(monitor.run())
    await monitor.wait_started()

    logger.info("Server ready")
    yield

    logger.info("Shutting down...")
    monitor.stop()
    capture.stop()
    await app.state.monitor_task


app = FastAPI(title="Video Chat with AI", lifespan=lifespan)

# Static files for web UI (Step 7)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# --- Endpoints ---


@app.get("/")
async def root():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Video Chat with AI - API running. Web UI: see Step 7."}


@app.get("/api/status", response_model=StatusResponse)
async def get_status(request: Request):
    monitor = request.app.state.monitor
    capture = request.app.state.capture
    window = request.app.state.window
    return StatusResponse(
        model_loaded=True,
        capture_running=capture.is_running,
        monitor_mode=monitor.mode,
        instruction=monitor.instruction,
        cycle_count=monitor.cycle_count,
        frames_buffered=window.count,
    )


@app.post("/api/start")
async def start_capture(body: StartRequest, request: Request):
    capture = request.app.state.capture
    if capture.is_running:
        raise HTTPException(409, "Capture already running. Stop first.")
    try:
        capture.start(body.source)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"status": "started", "source": body.source}


@app.post("/api/stop")
async def stop_capture(request: Request):
    capture = request.app.state.capture
    monitor = request.app.state.monitor
    capture.stop()
    monitor.set_instruction(None)
    return {"status": "stopped"}


@app.post("/api/instruction")
async def set_instruction(body: InstructionRequest, request: Request):
    monitor = request.app.state.monitor
    monitor.set_instruction(body.instruction)
    return {"status": "ok", "instruction": body.instruction}


@app.get("/api/stream")
async def stream_sse(request: Request):
    monitor = request.app.state.monitor

    async def event_generator():
        q = monitor.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                    if event is None:
                        break
                    elif isinstance(event, dict):
                        yield f"event: cycle_end\ndata: {json.dumps(event)}\n\n"
                    else:
                        # SSE data lines cannot contain newlines; split if needed
                        for line in event.split("\n"):
                            yield f"data: {line}\n"
                        yield "\n"
                except asyncio.TimeoutError:
                    # Keepalive comment to prevent proxy/browser timeout
                    yield ": keepalive\n\n"
        finally:
            monitor.unsubscribe(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/frame")
async def get_frame(request: Request):
    """Single frame snapshot (always real-time, no delay). For API consumers."""
    capture = request.app.state.capture
    frame = capture.latest_frame
    if frame is None:
        raise HTTPException(404, "No frame available")
    buf = io.BytesIO()
    frame.save(buf, format="JPEG", quality=FRAME_JPEG_QUALITY)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.get("/api/mjpeg")
async def mjpeg_stream(request: Request):
    """MJPEG video stream with adaptive delay for commentary sync.

    Reads from the display buffer in FrameCapture (native frame rate),
    NOT from the inference SlidingWindow (2 FPS). This gives smooth
    playback at the source's original frame rate.

    When STREAM_DELAY_INIT > 0, frames are served with an adaptive delay
    that tracks inference latency via EMA. When STREAM_DELAY_INIT == 0,
    frames are served in real-time (no sync).
    """
    capture = request.app.state.capture
    monitor = request.app.state.monitor
    # Match source FPS for smooth playback, fall back to MJPEG_FPS
    display_fps = capture.source_fps if capture.source_fps > 0 else MJPEG_FPS
    interval = 1.0 / display_fps

    async def generate():
        next_push = time.monotonic()
        while True:
            if await request.is_disconnected():
                break

            # Get JPEG from display buffer: delayed or real-time
            if STREAM_DELAY_INIT > 0:
                target_time = time.time() - monitor.target_delay
                jpeg = capture.get_display_jpeg(target_time)
            else:
                jpeg = capture.get_display_jpeg()

            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n" + jpeg + b"\r\n"
                )

            # Consistent timing at source frame rate
            next_push += interval
            sleep_for = next_push - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                next_push = time.monotonic()

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Entrypoint ---


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=SERVER_HOST, port=SERVER_PORT)
