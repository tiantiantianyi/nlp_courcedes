from pathlib import Path

from PIL import Image

from anima_search.app.mock_service import MockSearchService


def test_mock_search_is_deterministic_and_annotation_optional(tmp_path: Path):
    image_dir = tmp_path / "Val"
    image_dir.mkdir()
    for name in ("1.jpg", "2.jpg", "3.jpg"):
        Image.new("RGB", (8, 8)).save(image_dir / name)
    service = MockSearchService(tmp_path, image_dir, result_count=2)
    first = service.search("夜晚街道")
    second = service.search("夜晚街道")
    assert [item.image_id for item in first] == [item.image_id for item in second]
    assert len(first) == 2
    assert first[0].active_branches == ["mock"]
    assert service.annotations == {}


def test_mock_service_rejects_empty_directory(tmp_path: Path):
    image_dir = tmp_path / "Val"
    image_dir.mkdir()
    try:
        MockSearchService(tmp_path, image_dir)
    except ValueError as exc:
        assert "no supported images" in str(exc)
    else:
        raise AssertionError("empty mock directory must fail")
