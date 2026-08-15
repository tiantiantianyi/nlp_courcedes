from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

from PIL import Image

from anima_search.annotation.validation import extract_annotation_json, normalize_annotation_payload
from anima_search.schemas import ImageAnnotation, ManifestItem


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["image_id"] for line in path.read_text(encoding="utf-8").splitlines() if line}


def annotate_items(items: Iterable[ManifestItem], client: object, prompt: str, output_path: Path,
                   project_root: Path, prompt_version: str, retries: int = 2,
                   max_new_tokens: int = 1024) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    success = failed = 0
    failures_path = output_path.with_suffix(".failures.jsonl")
    for item in items:
        if not item.valid or item.image_id in completed:
            continue
        started = time.perf_counter()
        current_prompt = prompt
        error = ""
        for attempt in range(retries + 1):
            try:
                with Image.open(project_root / item.relative_path) as image:
                    raw = client.generate(image.copy(), current_prompt, max_new_tokens=max_new_tokens)
                payload = normalize_annotation_payload(extract_annotation_json(raw))
                metadata = getattr(client, "last_generation_metadata", {})
                payload.update(image_id=item.image_id, split=item.split, relative_path=item.relative_path,
                               sha256=item.sha256, duplicate_of=item.duplicate_of,
                               model_version=str(client.model_path), model_digest=getattr(client, "model_digest", ""),
                               prompt_version=prompt_version, prompt_sha256=prompt_sha256,
                               generation_parameters={
                                   "max_new_tokens": max_new_tokens,
                                   "do_sample": False,
                                   "repetition_penalty": 1.10,
                                   "max_image_pixels": getattr(client, "max_image_pixels", None),
                                   "input_width": metadata.get("input_width"),
                                   "input_height": metadata.get("input_height"),
                               },
                               peak_vram_bytes=metadata.get("peak_vram_bytes"),
                               elapsed_seconds=time.perf_counter() - started)
                annotation = ImageAnnotation.model_validate(payload)
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(annotation.model_dump_json() + "\n")
                success += 1
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                current_prompt = prompt + "\n上一次输出无效，请严格只输出满足字段要求的 JSON。"
                if attempt == retries:
                    with failures_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"image_id": item.image_id, "error": error}, ensure_ascii=False) + "\n")
                    failed += 1
    return success, failed
