from anima_search.runtime.model_manager import ModelManager
from anima_search.training.pairs import build_training_pairs


class Releasable:
    def __init__(self): self.released = False
    def unload(self): self.released = True


def test_model_manager_releases_qwen_before_sd():
    qwen = Releasable(); manager = ModelManager(lambda: qwen, lambda: object())
    with manager.qwen_session():
        pass
    with manager.sd_session():
        pass
    assert qwen.released


def test_training_pairs_reject_val_annotations():
    import pytest
    from anima_search.schemas import ImageAnnotation
    annotation = ImageAnnotation(image_id="val-1", split="Val", relative_path="Val/1.jpg", sha256="x",
        summary="x", scene="x", search_queries=["a", "b", "c"], generation_prompt="x",
        model_version="x", prompt_version="x")
    with pytest.raises(ValueError): build_training_pairs([annotation])


def test_training_pairs_include_explicit_negative_document():
    from anima_search.schemas import ImageAnnotation
    first = ImageAnnotation(image_id="train-1", split="Train", relative_path="Train/1.jpg", sha256="1",
        summary="雨夜城市", scene="城市", search_queries=["雨夜", "城市", "冷色"],
        generation_prompt="city", model_version="qwen", prompt_version="v1")
    second = ImageAnnotation(image_id="train-2", split="Train", relative_path="Train/2.jpg", sha256="2",
        summary="晴天城市", scene="城市", search_queries=["晴天", "街道", "暖色"],
        generation_prompt="city", model_version="qwen", prompt_version="v1")
    pair = build_training_pairs([first, second])[0]
    assert pair.negative and pair.negative_image_id != pair.image_id
