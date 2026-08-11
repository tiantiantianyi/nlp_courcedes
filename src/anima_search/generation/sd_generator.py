from __future__ import annotations

import json
from pathlib import Path

from anima_search.schemas import GenerationPrompts


class StableDiffusionGenerator:
    def __init__(self, model_path: str | Path, dtype: str = "float16", device: str = "cuda") -> None:
        self.model_path = Path(model_path)
        self.dtype_name = dtype
        self.device = device
        self.pipeline = None

    def load(self) -> None:
        if self.pipeline is not None:
            return
        import torch
        from diffusers import StableDiffusionPipeline
        dtype = torch.float32 if self.device == "cpu" else getattr(torch, self.dtype_name)
        load_options = {
            "torch_dtype": dtype,
            "local_files_only": True,
        }
        if self.device != "cpu" and self.dtype_name == "float16":
            load_options.update({"variant": "fp16", "use_safetensors": True})
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            **load_options,
        )
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_slicing()
        if self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(self.device)

    def unload(self) -> None:
        self.pipeline = None

    def generate(self, prompts: GenerationPrompts, output_dir: Path, seed: int,
                 width: int = 512, height: int = 512, steps: int = 30,
                 guidance_scale: float = 7.5) -> Path:
        import torch
        self.load()
        output_dir.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image = self.pipeline(prompt=prompts.positive_prompt,
            negative_prompt=prompts.negative_prompt, width=width, height=height,
            num_inference_steps=steps, guidance_scale=guidance_scale,
            generator=generator).images[0]
        output_path = output_dir / f"generated-{seed}.png"
        image.save(output_path)
        output_path.with_suffix(".json").write_text(json.dumps({
            "source": "generated", "positive_prompt": prompts.positive_prompt,
            "negative_prompt": prompts.negative_prompt, "seed": seed, "width": width,
            "height": height, "steps": steps, "guidance_scale": guidance_scale,
            "model_path": str(self.model_path)}, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path
