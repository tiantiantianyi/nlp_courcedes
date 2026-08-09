from __future__ import annotations

import json

import numpy as np
import pytest

from anima_search.indexing.bm25_index import BM25Index
from anima_search.indexing.image_vector_index import ImageVectorIndex, _projected_features
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
