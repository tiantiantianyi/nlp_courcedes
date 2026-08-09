from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from anima_search.provenance import model_directory_fingerprint


class QwenVLClient:
    def __init__(self, model_path: str | Path, dtype: str = "float16", device: str = "cuda",
                 max_image_pixels: int = 1024 * 1024) -> None:
        self.model_path = Path(model_path)
        self.dtype_name = dtype
        self.device = device
        self.max_image_pixels = max_image_pixels
        self.model_digest = model_directory_fingerprint(self.model_path)
        self.last_generation_metadata: dict[str, int] = {}
        self.model: Any = None
        self.processor: Any = None

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        pixels = rgb.width * rgb.height
        if pixels <= self.max_image_pixels:
            return rgb
        scale = math.sqrt(self.max_image_pixels / pixels)
        size = (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale)))
        return rgb.resize(size, Image.Resampling.LANCZOS)

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        dtype = torch.float32 if self.device == "cpu" else getattr(torch, self.dtype_name)
        kwargs = {"torch_dtype": dtype, "local_files_only": True}
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_path, **kwargs).eval()
        if self.device != "cuda":
            self.model.to(self.device)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 1024) -> str:
        self.load()
        import torch
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        original_size = image.size
        rgb = self._prepare_image(image)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": rgb}, {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[rgb], return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.10,
        )
        trimmed = [tokens[len(source):] for source, tokens in zip(inputs.input_ids, output)]
        self.last_generation_metadata = {
            "original_width": original_size[0],
            "original_height": original_size[1],
            "input_width": rgb.width,
            "input_height": rgb.height,
        }
        if torch.cuda.is_available() and self.device == "cuda":
            self.last_generation_metadata["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def generate_text(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.load()
        import torch
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [tokens[len(source):] for source, tokens in zip(inputs.input_ids, output)]
        self.last_generation_metadata = {"peak_vram_bytes": int(torch.cuda.max_memory_allocated())} \
            if torch.cuda.is_available() and self.device == "cuda" else {}
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
