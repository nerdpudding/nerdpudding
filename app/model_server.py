import os
import logging
from pathlib import Path
from typing import Generator

# Environment must be set BEFORE importing transformers/torch
from app.config import CUDA_VISIBLE_DEVICES, MODEL_PATH

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["HF_HOME"] = str(Path(MODEL_PATH).parent / ".hf_cache")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

from app.config import (  # noqa: E402
    MAX_INP_LENGTH,
    MAX_NEW_TOKENS,
    MAX_SLICE_NUMS,
    SUPPRESS_TOKENS,
)

logger = logging.getLogger(__name__)


class ModelServer:
    """Loads MiniCPM-o 4.5 in vision-only mode and provides inference."""

    def __init__(self, model_path: str = MODEL_PATH):
        hf_cache = os.environ["HF_HOME"]
        os.makedirs(hf_cache, exist_ok=True)
        logger.info(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}, HF_HOME={hf_cache}")
        logger.info(f"Loading model from {model_path} (vision-only mode)")

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.init_audio = False
        config.init_tts = False
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
        self.is_awq = is_awq

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        logger.info("Model loaded successfully")

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
