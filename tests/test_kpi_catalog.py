from types import SimpleNamespace

from finground.kpi_catalog import KNOWLEDGE_STATE_KEY, KPI_CATALOG, get_kpi_knowledge


def test_catalog_contains_complete_canonical_records() -> None:
    assert len(KPI_CATALOG) == 31
    for key, item in KPI_CATALOG.items():
        assert item.key == key
        assert item.definition
        assert item.accepted_labels
        assert item.rejected_labels
        assert item.preferred_statements
        assert item.retrieval_hints
        assert item.normalization_rule
        assert item.cautions


def test_knowledge_tool_requires_exact_key_and_records_scope() -> None:
    context = SimpleNamespace(state={})

    success = get_kpi_knowledge("revenue", context)
    unknown = get_kpi_knowledge("revenues", context)

    assert success["knowledge"]["key"] == "revenue"
    assert context.state[KNOWLEDGE_STATE_KEY] == "revenue"
    assert unknown["status"] == "error"
    assert "revenue" in unknown["supported_kpis"]
