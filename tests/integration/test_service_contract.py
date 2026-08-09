def test_service_contract_is_explicit():
    from anima_search.app.service import SearchService
    assert all(hasattr(SearchService, name) for name in (
        "search", "answer_about_image", "write_content", "generate_image"))
