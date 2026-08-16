from anima_search.config import load_config
from scripts.annotate_images import generation_prompt_version


def test_default_config_keeps_generation_version_separate_from_canonical_import() -> None:
    config = load_config("configs/default.yaml")

    assert config["annotation"]["prompt_version"] == "qwen35-canonical-v1.3"
    assert generation_prompt_version(config) == "caption_verified_v4"


def test_generation_version_falls_back_for_existing_configs() -> None:
    config = {"annotation": {"prompt_version": "legacy-v1"}}

    assert generation_prompt_version(config) == "legacy-v1"
