from ingestion.entity_extractor import _parse_response, ENTITY_KEYS


def test_parse_response_valid_json():
    raw = '{"equipment_tags": ["P-101"], "dates": ["2026-03-14"], "personnel": [], "regulatory_refs": [], "document_refs": []}'
    result = _parse_response(raw)
    assert result["equipment_tags"] == ["P-101"]
    assert result["dates"] == ["2026-03-14"]
    assert set(result.keys()) == set(ENTITY_KEYS)


def test_parse_response_strips_markdown_fences():
    raw = '```json\n{"equipment_tags": ["HX-204"], "dates": [], "personnel": [], "regulatory_refs": [], "document_refs": []}\n```'
    result = _parse_response(raw)
    assert result["equipment_tags"] == ["HX-204"]


def test_parse_response_invalid_json_returns_empty_lists():
    result = _parse_response("not json at all")
    assert all(result[key] == [] for key in ENTITY_KEYS)
