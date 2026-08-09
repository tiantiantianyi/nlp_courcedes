from PIL import Image

from anima_search.data.manifest import scan_split


def test_manifest_supports_image_directory_next_to_project(tmp_path):
    project = tmp_path / "anima"
    image_dir = tmp_path / "Train"
    project.mkdir()
    image_dir.mkdir()
    Image.new("RGB", (8, 8), color="white").save(image_dir / "0.jpg")

    items = scan_split(image_dir, "Train", project)

    assert items[0].relative_path == "../Train/0.jpg"
