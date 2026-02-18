import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

# Environment must be set BEFORE importing transformers/torch
from app.config import CUDA_VISIBLE_DEVICES, MODEL_PATH

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["HF_HOME"] = str(Path(MODEL_PATH).parent / ".hf_cache")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

from app.config import (  # noqa: E402
    ENABLE_TTS,
    MAX_INP_LENGTH,
    MAX_NEW_TOKENS,
    MAX_SLICE_NUMS,
    REF_AUDIO_PATH,
    SUPPRESS_TOKENS,
    TTS_FLOAT16,
    TTS_MAX_NEW_TOKENS,
    TTS_MODEL_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """One chunk from streaming inference. Audio is None when TTS is disabled."""

    text: str
    audio: Optional[torch.Tensor]  # (1, N) float32 at 24kHz, or None
    is_last: bool


class ModelServer:
    """Loads MiniCPM-o 4.5 and provides text and text+audio inference."""

    def __init__(self, model_path: str = MODEL_PATH, enable_tts: bool = ENABLE_TTS):
        hf_cache = os.environ["HF_HOME"]
        os.makedirs(hf_cache, exist_ok=True)
        logger.info(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}, HF_HOME={hf_cache}")
        logger.info(f"Loading model from {model_path} (TTS={'enabled' if enable_tts else 'disabled'})")

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.init_audio = False
        config.init_tts = enable_tts
        config.init_vision = True

        is_awq = hasattr(config, "quantization_config")
        dtype = torch.float16 if is_awq else torch.bfloat16
        logger.info(f"Model type: {'AWQ INT4' if is_awq else 'BF16'}, dtype: {dtype}")

        self.model = AutoModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=dtype,
        )
        self.model.eval().cuda()
        try:
            self.model = torch.compile(self.model)
            logger.info("torch.compile() enabled")
        except Exception as e:
            logger.warning(f"torch.compile() failed, using eager mode: {e}")
        self.is_awq = is_awq
        self.tts_enabled = enable_tts

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        if enable_tts:
            self._init_tts()

        self._session_counter = 0
        logger.info("Model loaded successfully")

    def _init_tts(self) -> None:
        """Initialize TTS vocoder and reference audio cache."""
        import librosa

        logger.info(f"Initializing TTS vocoder from {TTS_MODEL_DIR}")

        # AWQ model's init_tts() defaults to CosyVoice2 (incompatible with
        # streaming_generate). Must pass streaming=True to use Token2wav.
        # BF16 model always uses Token2wav (no streaming parameter).
        tts_kwargs = {
            "model_dir": TTS_MODEL_DIR,
            "enable_float16": TTS_FLOAT16,
        }
        if self.is_awq:
            tts_kwargs["streaming"] = True

        self.model.init_tts(**tts_kwargs)
        self.model.reset_session(reset_token2wav_cache=True)

        # Load reference audio and prime the vocoder cache
        logger.info(f"Loading reference audio from {REF_AUDIO_PATH}")
        ref_audio, _ = librosa.load(REF_AUDIO_PATH, sr=16000, mono=True)
        self.model.init_token2wav_cache(prompt_speech_16k=ref_audio)

        logger.info("TTS initialized successfully")

    def infer(
        self,
        frames: list[Image.Image],
        instruction: str,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """Run inference on a list of frames with an instruction.

        Args:
            frames: List of PIL Images (RGB) to analyze.
            instruction: User instruction, e.g. "describe what's happening".
            stream: If True, yield text chunks. If False, yield a single result.

        Yields:
            Text chunks from the model.
        """
        msgs = [{"role": "user", "content": frames + [instruction]}]

        params = {
            "image": None,
            "msgs": msgs,
            "tokenizer": self.tokenizer,
            "use_image_id": False,
            "max_slice_nums": MAX_SLICE_NUMS,
            "max_inp_length": MAX_INP_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "suppress_tokens": SUPPRESS_TOKENS,
        }

        if stream:
            params["stream"] = True
            params["num_beams"] = 1
            params["do_sample"] = True

            streamer = self.model.chat(**params)
            for chunk in streamer:
                cleaned = chunk.replace("<|im_end|>", "")
                if cleaned:
                    yield cleaned
        else:
            response = str(self.model.chat(**params))
            yield response.replace("<|im_end|>", "")

    def infer_with_audio(
        self,
        frames: list[Image.Image],
        instruction: str,
    ) -> Generator[InferenceResult, None, None]:
        """Streaming inference with TTS audio output.

        When TTS is disabled, falls back to text-only via infer().

        Args:
            frames: List of PIL Images (RGB) to analyze.
            instruction: User instruction text.

        Yields:
            InferenceResult with text chunks and optional audio waveform.
        """
        if not self.tts_enabled:
            for chunk in self.infer(frames, instruction, stream=True):
                yield InferenceResult(text=chunk, audio=None, is_last=False)
            yield InferenceResult(text="", audio=None, is_last=True)
            return

        self._session_counter += 1
        sid = str(self._session_counter)
        effective_max_tokens = TTS_MAX_NEW_TOKENS if TTS_MAX_NEW_TOKENS > 0 else MAX_NEW_TOKENS

        msg = {"role": "user", "content": frames + [instruction]}
        self.model.streaming_prefill(
            session_id=sid,
            msgs=[msg],
            max_slice_nums=MAX_SLICE_NUMS,
            use_tts_template=True,
            is_last_chunk=True,
        )

        for wav_chunk, text_chunk in self.model.streaming_generate(
            session_id=sid,
            generate_audio=True,
            use_tts_template=True,
            max_new_tokens=effective_max_tokens,
            do_sample=True,
        ):
            if wav_chunk is None and text_chunk is None:
                break
            yield InferenceResult(
                text=text_chunk or "",
                audio=wav_chunk,
                is_last=False,
            )
        yield InferenceResult(text="", audio=None, is_last=True)
