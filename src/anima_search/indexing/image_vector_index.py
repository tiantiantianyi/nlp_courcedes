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
    if not np.all(np.isfinite(matrix)):
        raise ValueError("encoder returned a non-finite vector")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)):
        raise ValueError("encoder vector norm is non-finite")
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


def _install_transformers_clip_loss_compatibility() -> None:
    """Restore the Transformers 4.x helper imported by Jina's remote code."""
    import torch
    import torch.nn.functional as functional
    import transformers.models.clip.modeling_clip as clip_module

    if hasattr(clip_module, "clip_loss"):
        return

    def contrastive_loss(logits):
        labels = torch.arange(len(logits), device=logits.device)
        return functional.cross_entropy(logits, labels)

    def clip_loss(similarity):
        return (
            contrastive_loss(similarity)
            + contrastive_loss(similarity.transpose(0, 1))
        ) / 2

    clip_module.contrastive_loss = contrastive_loss
    clip_module.clip_loss = clip_loss


def _repair_jina_vision_rope_buffers(model) -> bool:
    """Deterministically rebuild EVA RoPE buffers omitted from checkpoints."""
    import torch

    vision_model = getattr(model, "vision_model", None)
    rope = getattr(vision_model, "rope", None)
    if rope is None:
        return False

    config = model.config.vision_config
    num_heads = config.width // config.head_width
    half_head_dim = config.width // num_heads // 2
    sequence_length = (
        config.image_size // config.patch_size
        if config.intp_freq
        else config.pt_hw_seq_len
    )
    frequencies = 1.0 / (
        10000
        ** (
            torch.arange(0, half_head_dim, 2, dtype=torch.float32)
            / half_head_dim
        )
    )
    positions = (
        torch.arange(sequence_length, dtype=torch.float32)
        / sequence_length
        * config.pt_hw_seq_len
    )
    frequencies = torch.einsum("i,j->ij", positions, frequencies)
    frequencies = torch.repeat_interleave(frequencies, 2, dim=-1)
    height = frequencies[:, None, :].expand(-1, sequence_length, -1)
    width = frequencies[None, :, :].expand(sequence_length, -1, -1)
    frequencies = torch.cat((height, width), dim=-1).reshape(
        sequence_length * sequence_length,
        -1,
    )
    rope.freqs_cos = frequencies.cos()
    rope.freqs_sin = frequencies.sin()
    return True


class JinaClipV2Encoder:
    """Lazy jina-clip-v2 adapter with Matryoshka truncation."""

    MATRYOSHKA_DIMENSIONS = {32, 64, 128, 256, 512, 768, 1024}

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        dtype: str = "float16",
        *,
        truncate_dim: int | None = 512,
        local_files_only: bool = True,
    ) -> None:
        if truncate_dim is not None and truncate_dim not in self.MATRYOSHKA_DIMENSIONS:
            raise ValueError(
                "jina-clip-v2 truncate_dim must be one of "
                f"{sorted(self.MATRYOSHKA_DIMENSIONS)} or None"
            )
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.truncate_dim = truncate_dim
        self.local_files_only = local_files_only
        self.model_digest = model_directory_fingerprint(model_path)
        self.model = None

    def _load(self):
        if self.model is not None:
            return self.model
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for jina-clip-v2 but is not available")
        _install_transformers_clip_loss_compatibility()
        torch_dtype = (
            getattr(torch, self.dtype)
            if self.device.startswith("cuda")
            else torch.float32
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            dtype=torch_dtype,
        )
        _repair_jina_vision_rope_buffers(self.model)
        self.model.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            fix_mistral_regex=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        return self.model

    def encode_images(self, image_paths: list[Path], batch_size: int = 1) -> np.ndarray:
        model = self._load()
        batches: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            paths = [str(path.resolve()) for path in image_paths[start:start + batch_size]]
            features = model.encode_image(
                paths,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=self.device,
                normalize_embeddings=True,
                truncate_dim=self.truncate_dim,
            )
            batches.append(np.asarray(features, dtype=np.float32))
        if not batches:
            dimension = self.truncate_dim or 1024
            return np.empty((0, dimension), dtype=np.float32)
        return _normalized_matrix(np.concatenate(batches, axis=0))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            dimension = self.truncate_dim or 1024
            return np.empty((0, dimension), dtype=np.float32)
        features = model.encode_text(
            texts,
            task="retrieval.query",
            batch_size=min(32, len(texts)),
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True,
            truncate_dim=self.truncate_dim,
        )
        return _normalized_matrix(np.asarray(features, dtype=np.float32))

    def unload(self) -> None:
        self.model = None
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
                 encoder: object | None = None, encoder_type: str = "chinese_clip",
                 encoder_options: dict | None = None) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.annotation_version = annotation_version
        self.build_parameters = build_parameters or {}
        self.encoder = encoder
        self.encoder_type = encoder_type
        self.encoder_options = encoder_options or {}
        self.model_digest = getattr(encoder, "model_digest", model_directory_fingerprint(model_path))
        self.index = None
        self.image_ids: list[str] = []

    def _load_encoder(self):
        if self.encoder is None:
            if self.encoder_type == "chinese_clip":
                self.encoder = ChineseClipEncoder(
                    self.model_path,
                    self.device,
                    self.dtype,
                )
            elif self.encoder_type == "jina_clip_v2":
                self.encoder = JinaClipV2Encoder(
                    self.model_path,
                    self.device,
                    self.dtype,
                    **self.encoder_options,
                )
            else:
                raise ValueError(
                    "image encoder_type must be chinese_clip or jina_clip_v2"
                )
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
            "encoder_type": self.encoder_type,
            "encoder_options": self.encoder_options,
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
        encoder_type: str | None = None,
        encoder_options: dict | None = None,
    ) -> ImageVectorIndex:
        payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        instance = cls(
            model_path or payload["model_path"],
            device if device is not None else payload.get("device", "cuda"),
            dtype if dtype is not None else payload.get("dtype", "float16"),
            payload.get("annotation_version", ""),
            payload.get("build_parameters", {}),
            encoder=encoder,
            encoder_type=encoder_type or payload.get("encoder_type", "chinese_clip"),
            encoder_options=(
                encoder_options
                if encoder_options is not None
                else payload.get("encoder_options", {})
            ),
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
