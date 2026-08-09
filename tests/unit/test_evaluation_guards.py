from anima_search.indexing.documents import annotation_to_document
from anima_search.schemas import ImageAnnotation


def test_index_document_does_not_embed_generated_search_queries():
    item = ImageAnnotation(image_id="val-1", split="Val", relative_path="Val/1.jpg", sha256="x",
        summary="城市夜景", scene="城市", search_queries=["verbatim-a", "b", "c"],
        generation_prompt="city", model_version="qwen", prompt_version="v1")
    assert "verbatim-a" not in annotation_to_document(item)
