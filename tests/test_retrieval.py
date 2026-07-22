from finground.documents import Page
from finground.retrieval import rank_pages


def test_retrieval_recognizes_consolidated_statement_heading_word_order() -> None:
    pages = [
        Page(
            raw_index=0,
            text="Management discussion repeatedly mentions cost of goods sold in 2023.",
        ),
        Page(
            raw_index=1,
            text=(
                "Statements of Consolidated Income (in millions) 2023 "
                "Net sales 200 Cost of goods sold 120 Gross profit 80"
            ),
        ),
    ]

    hits = rank_pages(
        pages,
        "What was cost of revenue in 2023?",
        ["cost_of_revenue"],
        2023,
    )

    statement_hit = next(hit for hit in hits if hit.page.raw_index == 1)
    assert statement_hit.components["primary_statement"] == 4.0
    assert statement_hit.components["exact_phrase"] > 0
    assert hits[0].page.raw_index == 1
