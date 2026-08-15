from contextlib import contextmanager
from pathlib import Path

import pytest

from anima_search.app.service import SearchService
from anima_search.schemas import SearchQuery


class Parser:
    def __init__(self) -> None:
        self.client = "not-called"

    def parse(self, query: str, client=None):
        self.client = client
        return SearchQuery(raw_text=query)


class Searcher:
    def search(self, query, candidate_count, result_count):
        return []


class Manager:
    def __init__(self) -> None:
        self.qwen = object()
        self.unloaded = False

    @contextmanager
    def qwen_session(self):
        yield self.qwen

    def unload_all(self):
        self.unloaded = True


def service(tmp_path: Path, backend: str) -> tuple[SearchService, Parser, Manager]:
    qwen_path = tmp_path / "qwen"
    qwen_path.mkdir()
    parser = Parser()
    manager = Manager()
    instance = SearchService(
        {
            "project_root": str(tmp_path),
            "models": {"qwen_vl": str(qwen_path)},
            "retrieval": {
                "query_parser_backend": backend,
                "candidate_count": 5,
                "result_count": 3,
                "rerank_count": 3,
            },
        },
        parser,
        Searcher(),
        manager,
        {},
        "",
        "",
        "",
    )
    return instance, parser, manager


def test_local_qwen_backend_passes_managed_client(tmp_path: Path):
    instance, parser, manager = service(tmp_path, "local_qwen")
    instance.search("雨夜城市")
    assert parser.client is manager.qwen
    assert manager.unloaded


def test_api_backend_uses_parser_owned_client_without_local_qwen(tmp_path: Path):
    instance, parser, manager = service(tmp_path, "openai_compatible")
    instance.search("雨夜城市")
    assert parser.client is None
    assert not manager.unloaded


def test_unknown_parser_backend_is_rejected(tmp_path: Path):
    instance, _, _ = service(tmp_path, "mystery")
    with pytest.raises(ValueError, match="query_parser_backend"):
        instance.search("雨夜城市")
