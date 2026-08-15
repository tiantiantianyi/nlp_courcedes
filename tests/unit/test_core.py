from anima_search.annotation.validation import (
    extract_annotation_json,
    extract_json_object,
    normalize_annotation_payload,
)
from anima_search.retrieval.fusion import reciprocal_rank_fusion
from anima_search.retrieval.search import query_to_document
from anima_search.schemas import SearchQuery


def test_extract_json_object_ignores_surrounding_text():
    assert extract_json_object('prefix {"summary": "ok"} suffix')["summary"] == "ok"


def test_annotation_payload_normalizes_vlm_scalar_variants():
    payload = normalize_annotation_payload({
        "attributes": {"lighting": "逆光"},
        "style": "摄影",
        "colors": ["橙色", "橙色"] + [str(index) for index in range(12)],
        "ocr_text": "",
        "_uncertainty": "无法确认地点",
        "scene": ["公路", "黄昏"],
    })
    assert payload["attributes"] == ["lighting:逆光"]
    assert payload["style"] == ["摄影"]
    assert payload["colors"] == ["橙色"] + [str(index) for index in range(9)]
    assert payload["ocr_text"] == []
    assert payload["uncertainty"] == ["无法确认地点"]
    assert payload["scene"] == "公路、黄昏"


def test_annotation_json_repairs_truncated_final_ocr_array():
    raw = '{"summary":"ok","search_queries":["a","b","c"],"ocr_text":["A","A","B","C'
    payload = extract_annotation_json(raw)
    assert payload["summary"] == "ok"
    assert payload["search_queries"] == ["a", "b", "c"]
    assert payload["ocr_text"] == ["A", "B"]


def test_qwen_client_caps_image_pixels_without_changing_aspect_ratio():
    from PIL import Image
    from anima_search.annotation.qwen_client import QwenVLClient

    client = QwenVLClient("unused", max_image_pixels=10_000)
    prepared = client._prepare_image(Image.new("RGB", (400, 200)))
    assert prepared.width * prepared.height <= 10_000
    assert abs(prepared.width / prepared.height - 2.0) < 0.02


def test_rrf_rewards_multiple_branches():
    results = reciprocal_rank_fusion({"a": [("x", 1.0), ("y", 0.5)], "b": [("y", 1.0)]})
    assert results[0][0] == "y"


def test_search_query_defaults_to_empty_filters():
    assert SearchQuery(raw_text="雨夜城市").excluded_terms == []


def test_structured_query_fields_enter_retrieval_text():
    query = SearchQuery(raw_text="找照片", objects=["汽车"], scene=["城市"], colors=["冷色"])
    document = query_to_document(query)
    assert "主体:汽车" in document and "场景:城市" in document and "颜色:冷色" in document
