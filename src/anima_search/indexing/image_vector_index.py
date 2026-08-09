from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anima_search.indexing.faiss_io import read_faiss_index, write_faiss_index
from anima_search.provenance import model_directory_fingerprint


def _normalized_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("encoder output must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("encoder returned a zero vector")
    return matrix / norms


def _projected_features(output):
    """Return projected features across Transformers Chinese-CLIP APIs."""
    return output.pooler_output if hasattr(output, "pooler_output") else output


class ChineseClipEncoder:
    """Lazy local Chinese-CLIP adapter used by the image retrieval branch."""

    def __init__(self, model_path: str | Path, device: str = "cuda", dtype: str = "float16") -> None:
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.model_digest = model_directory_fingerprint(model_path)
        self.model = None
        self.processor = None

    def _load(self):
        if self.model is not None:
            return self.model, self.processor
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Chinese-CLIP but is not available")
        torch_dtype = getattr(torch, self.dtype) if self.device.startswith("cuda") else torch.float32
        self.processor = ChineseCLIPProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = ChineseCLIPModel.from_pretrained(
            self.model_path, local_files_only=True, torch_dtype=torch_dtype
        ).to(self.device)
        self.model.eval()
        return self.model, self.processor

    @staticmethod
    def _move_inputs(inputs: dict, device: str) -> dict:
        return {name: value.to(device) if hasattr(value, "to") else value for name, value in inputs.items()}

    def encode_images(self, image_paths: list[Path], batch_size: int = 8) -> np.ndarray:
        import torch
        from PIL import Image

        model, processor = self._load()
        batches: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            images = []
            for image_path in image_paths[start:start + batch_size]:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB").copy())
            inputs = self._move_inputs(processor(images=images, return_tensors="pt"), self.device)
            with torch.inference_mode():
                features = _projected_features(model.get_image_features(**inputs))
            batches.append(features.float().cpu().numpy())
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return _normalized_matrix(np.concatenate(batches, axis=0))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        model, processor = self._load()
        inputs = self._move_inputs(processor(text=texts, padding=True, return_tensors="pt"), self.device)
        with torch.inference_mode():
            features = _projected_features(model.get_text_features(**inputs))
        return _normalized_matrix(features.float().cpu().numpy())

    def unload(self) -> None:
        self.model = None
        self.processor = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class ImageVectorIndex:
    def __init__(self, model_path: str | Path, device: str = "cuda", dtype: str = "float16",
                 annotation_version: str = "", build_parameters: dict | None = None,
                 encoder: object | None = None) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.annotation_version = annotation_version
        self.build_parameters = build_parameters or {}
        self.encoder = encoder
        self.model_digest = getattr(encoder, "model_digest", model_directory_fingerprint(model_path))
        self.index = None
        self.image_ids: list[str] = []

    def _load_encoder(self):
        if self.encoder is None:
            self.encoder = ChineseClipEncoder(self.model_path, self.device, self.dtype)
        return self.encoder

    def build(self, image_ids: list[str], image_paths: list[Path], batch_size: int = 8) -> None:
        import faiss

        if len(image_ids) != len(image_paths):
            raise ValueError("image_ids and image_paths must have the same length")
        if not image_ids:
            raise ValueError("cannot build an empty image vector index")
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("image_ids must be unique")
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"image files are missing: {missing[:3]}")
        vectors = _normalized_matrix(self._load_encoder().encode_images(image_paths, batch_size=batch_size))
        if vectors.shape[0] != len(image_ids):
            raise ValueError("encoder returned a different number of image vectors")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.image_ids = list(image_ids)

    def search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("image vector index has not been built or loaded")
        if limit <= 0:
            return []
        vector = _normalized_matrix(self._load_encoder().encode_texts([query]))
        if vector.shape[1] != self.index.d:
            raise ValueError("query vector dimension does not match image index")
        scores, indices = self.index.search(vector, min(limit, len(self.image_ids)))
        return [(self.image_ids[int(index)], float(score))
                for score, index in zip(scores[0], indices[0]) if index >= 0]

    def save(self, directory: Path) -> None:
        if self.index is None:
            raise RuntimeError("cannot save an image vector index before build")
        directory.mkdir(parents=True, exist_ok=True)
        write_faiss_index(self.index, directory / "vectors.faiss")
        metadata = {
            "image_ids": self.image_ids,
            "model_path": self.model_path,
            "device": self.device,
            "dtype": self.dtype,
            "annotation_version": self.annotation_version,
            "model_digest": self.model_digest,
            "build_parameters": self.build_parameters,
            "dimension": self.index.d,
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        encoder: object | None = None,
        *,
        model_path: str | Path | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> ImageVectorIndex:
        payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        instance = cls(
            model_path or payload["model_path"],
            device if device is not None else payload.get("device", "cuda"),
            dtype if dtype is not None else payload.get("dtype", "float16"),
            payload.get("annotation_version", ""),
            payload.get("build_parameters", {}),
            encoder=encoder,
        )
        instance.model_digest = payload.get("model_digest", instance.model_digest)
        instance.image_ids = payload["image_ids"]
        instance.index = read_faiss_index(directory / "vectors.faiss")
        if instance.index.ntotal != len(instance.image_ids):
            raise ValueError("image vector index count does not match metadata image_ids")
        if payload.get("dimension") not in (None, instance.index.d):
            raise ValueError("image vector index dimension does not match metadata")
        return instance

    def unload_encoder(self) -> None:
        if self.encoder is not None and hasattr(self.encoder, "unload"):
            self.encoder.unload()
        self.encoder = None
