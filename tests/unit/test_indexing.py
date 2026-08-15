from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anima_search.indexing.bm25_index import BM25Index
from anima_search.indexing.image_vector_index import (
    ImageVectorIndex,
    JinaClipV2Encoder,
    _projected_features,
    _repair_jina_vision_rope_buffers,
)
from anima_search.indexing.index_manifest import (
    load_index_manifest,
    validate_index_manifest,
    write_index_manifest,
)
from anima_search.indexing.vector_index import VectorIndex


class FakeTextEncoder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "雨夜" in text else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class FakeImageEncoder:
    model_digest = "fake-image-model"

    def encode_images(self, image_paths, batch_size=8):
        return np.asarray([
            [1.0, 0.0] if path.stem == "first" else [0.0, 1.0]
            for path in image_paths
        ], dtype=np.float32)

    def encode_texts(self, texts):
        return np.asarray([
            [1.0, 0.0] if "第一" in text else [0.0, 1.0]
            for text in texts
        ], dtype=np.float32)


class FeatureOutput:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class FakeJinaModel:
    def __init__(self):
        self.image_options = None
        self.text_options = None

    def encode_image(self, images, **kwargs):
        self.image_options = kwargs
        return np.asarray([[1.0, 0.0] for _ in images], dtype=np.float32)

    def encode_text(self, texts, **kwargs):
        self.text_options = kwargs
        return np.asarray([[0.0, 1.0] for _ in texts], dtype=np.float32)


def test_chinese_clip_feature_output_compatibility():
    direct = object()
    wrapped = object()
    assert _projected_features(direct) is direct
    assert _projected_features(FeatureOutput(wrapped)) is wrapped


def test_bm25_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same length"):
        BM25Index(["a"], ["one", "two"])


def test_text_vector_index_round_trip_with_fake_encoder(tmp_path):
    index = VectorIndex("missing-model", encoder=FakeTextEncoder())
    index.build(["rain", "sun"], ["雨夜城市", "晴天公园"], batch_size=2)
    assert index.search("雨夜", 1)[0][0] == "rain"

    output = tmp_path / "text"
    index.save(output)
    restored = VectorIndex.load(output, encoder=FakeTextEncoder())
    assert restored.search("晴天", 1)[0][0] == "sun"

    relocated = VectorIndex.load(
        output, encoder=FakeTextEncoder(), model_path="models/relocated-bge", device="cpu"
    )
    assert relocated.model_path == "models/relocated-bge"
    assert relocated.device == "cpu"


def test_image_vector_index_round_trip_with_fake_encoder(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"fixture")
    second.write_bytes(b"fixture")
    index = ImageVectorIndex("missing-model", encoder=FakeImageEncoder())
    index.build(["first", "second"], [first, second], batch_size=1)
    assert index.search("第一张", 1)[0][0] == "first"

    output = tmp_path / "image"
    index.save(output)
    restored = ImageVectorIndex.load(output, encoder=FakeImageEncoder())
    assert restored.search("第二张", 1)[0][0] == "second"

    relocated = ImageVectorIndex.load(
        output,
        encoder=FakeImageEncoder(),
        model_path="models/relocated-chinese-clip",
        device="cpu",
        dtype="float32",
    )
    assert relocated.model_path == "models/relocated-chinese-clip"
    assert relocated.device == "cpu"
    assert relocated.dtype == "float32"


def test_image_vector_index_rejects_non_finite_vectors(tmp_path):
    class NonFiniteEncoder(FakeImageEncoder):
        def encode_images(self, image_paths, batch_size=8):
            return np.asarray([[np.nan, 1.0]], dtype=np.float32)

    image = tmp_path / "invalid.jpg"
    image.write_bytes(b"fixture")
    index = ImageVectorIndex("missing-model", encoder=NonFiniteEncoder())

    with pytest.raises(ValueError, match="non-finite"):
        index.build(["invalid"], [image])


def test_image_index_persists_jina_encoder_metadata(tmp_path):
    image = tmp_path / "first.jpg"
    image.write_bytes(b"fixture")
    index = ImageVectorIndex(
        str(tmp_path),
        encoder=FakeImageEncoder(),
        encoder_type="jina_clip_v2",
        encoder_options={"truncate_dim": 512, "local_files_only": True},
    )
    index.build(["first"], [image])
    output = tmp_path / "jina-index"
    index.save(output)

    restored = ImageVectorIndex.load(output, encoder=FakeImageEncoder())

    assert restored.encoder_type == "jina_clip_v2"
    assert restored.encoder_options["truncate_dim"] == 512


def test_jina_adapter_passes_retrieval_task_and_matryoshka_dimension(tmp_path):
    model = FakeJinaModel()
    encoder = JinaClipV2Encoder(
        tmp_path,
        device="cpu",
        dtype="float32",
        truncate_dim=64,
    )
    encoder.model = model

    image_vectors = encoder.encode_images([tmp_path / "one.jpg"], batch_size=1)
    text_vectors = encoder.encode_texts(["雨夜城市"])

    assert image_vectors.shape == (1, 2)
    assert text_vectors.shape == (1, 2)
    assert model.image_options["truncate_dim"] == 64
    assert model.text_options["truncate_dim"] == 64
    assert model.text_options["task"] == "retrieval.query"


def test_jina_adapter_rejects_untrained_truncation_dimension(tmp_path):
    with pytest.raises(ValueError, match="truncate_dim"):
        JinaClipV2Encoder(tmp_path, truncate_dim=100)


def test_jina_rope_buffers_are_rebuilt_deterministically():
    class FakeRope(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("freqs_cos", torch.full((1, 1), torch.nan))
            self.register_buffer("freqs_sin", torch.full((1, 1), torch.nan))

    rope = FakeRope()
    model = SimpleNamespace(
        vision_model=SimpleNamespace(rope=rope),
        config=SimpleNamespace(
            vision_config=SimpleNamespace(
                width=1024,
                head_width=64,
                image_size=512,
                patch_size=14,
                intp_freq=True,
                pt_hw_seq_len=16,
            )
        ),
    )

    assert _repair_jina_vision_rope_buffers(model) is True
    assert rope.freqs_cos.shape == (36 * 36, 64)
    assert torch.isfinite(rope.freqs_cos).all()
    assert torch.isfinite(rope.freqs_sin).all()
    torch.testing.assert_close(
        rope.freqs_cos.square() + rope.freqs_sin.square(),
        torch.ones_like(rope.freqs_cos),
    )


def test_manifest_rejects_annotation_id_mismatch(tmp_path):
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(json.dumps({"image_id": "a"}) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    write_index_manifest(
        manifest_path,
        split="val",
        image_ids=["a", "b"],
        annotation_path=annotations,
        annotation_version="v1",
        branches={"text": {"record_count": 2}},
    )
    manifest = load_index_manifest(manifest_path)
    validate_index_manifest(manifest, ["a", "b"], {"text": ["a", "b"]})
    with pytest.raises(ValueError, match="image IDs"):
        validate_index_manifest(manifest, ["b", "a"], {"text": ["b", "a"]})
