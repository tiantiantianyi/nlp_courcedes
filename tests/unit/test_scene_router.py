from __future__ import annotations

import numpy as np
import pytest

from anima_search.routing.scene_router import SceneDefinition, SceneRouter


class FakeEncoder:
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        vectors = {
            "室内": [1.0, 0.0],
            "房间": [1.0, 0.0],
            "自然": [0.0, 1.0],
            "风景": [0.0, 1.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def config() -> dict[str, object]:
    return {
        "categories": {
            "indoor": {
                "label": "室内",
                "prompts": ["室内", "房间"],
                "prompt_suffix": "核对家具。",
            },
            "nature": {
                "label": "自然",
                "prompts": ["自然", "风景"],
                "prompt_suffix": "核对植被。",
            },
        }
    }


def test_scene_router_routes_vectors_and_returns_top_scores():
    router = SceneRouter.from_config(FakeEncoder(), config())
    routes = router.route_vectors(
        ["val-1", "val-2"],
        np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
        top_n=2,
    )
    assert [route.category for route in routes] == ["indoor", "nature"]
    assert routes[0].top_scores[0][0] == "indoor"
    assert routes[1].score > routes[1].top_scores[1][1]


def test_scene_router_appends_category_prompt():
    router = SceneRouter.from_config(FakeEncoder(), config())
    route = router.route_vectors(["val-1"], np.asarray([[1.0, 0.0]]), top_n=1)[0]
    prompt = router.annotation_prompt("基础标注要求。", route)
    assert "场景路由：室内" in prompt
    assert "核对家具" in prompt


def test_scene_router_validates_category_and_vector_contracts():
    with pytest.raises(ValueError, match="at least one text prompt"):
        SceneDefinition.from_mapping(
            "empty",
            {"label": "空", "prompts": [], "prompt_suffix": "x"},
        )
    router = SceneRouter.from_config(FakeEncoder(), config())
    with pytest.raises(ValueError, match="count"):
        router.route_vectors(["only-one"], np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError, match="top_n"):
        router.route_vectors(["one"], np.asarray([[1.0, 0.0]]), top_n=3)
