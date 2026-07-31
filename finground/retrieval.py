"""Explainable retrieval for table-heavy financial pages."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from finground.documents import Page
from finground.kpis import KPI_ALIASES

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "did",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "were",
    "with",
}
PRIMARY_STATEMENT_TERMS = (
    "consolidated statements",
    "consolidated statement",
    "consolidated balance sheets",
    "statements of consolidated income",
    "statement of consolidated income",
    "statements of consolidated operations",
    "statements of consolidated cash flows",
    "statement of consolidated cash flows",
    "balance sheets",
    "income statements",
    "statements of income",
    "statements of operations",
    "statements of cash flows",
)
WEAK_SECTION_TERMS = ("table of contents", "financial highlights", "non-gaap", "adjusted ebitda")


@dataclass(frozen=True, slots=True)
class SearchHit:
    page: Page
    score: float
    components: dict[str, float]


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def expand_search_phrases(phrases: list[str]) -> list[str]:
    """Expand canonical KPI keys into report labels and remove duplicates."""
    expanded: list[str] = []
    seen: set[str] = set()
    for raw_phrase in phrases[:12]:
        phrase = raw_phrase.strip()
        if not phrase:
            continue
        canonical = phrase.casefold().replace("-", "_").replace(" ", "_")
        aliases = KPI_ALIASES.get(canonical)
        candidates = (phrase.replace("_", " "), *(aliases or ()))
        for candidate in candidates:
            key = candidate.casefold()
            if key not in seen:
                expanded.append(candidate)
                seen.add(key)
    return expanded[:48]


def rank_pages(
    pages: list[Page],
    query: str,
    phrases: list[str],
    year: int | None,
    *,
    preferred_page_numbers: frozenset[int] = frozenset(),
) -> list[SearchHit]:
    """Rank pages with BM25 plus transparent financial-structure features."""
    if not pages:
        return []
    expanded_phrases = expand_search_phrases(phrases)
    query_tokens = tokenize(" ".join((query, *expanded_phrases)))
    query_counts = Counter(query_tokens)
    document_tokens = [tokenize(page.text) for page in pages]
    document_frequencies: Counter[str] = Counter()
    for tokens in document_tokens:
        document_frequencies.update(set(tokens))
    average_length = sum(len(tokens) for tokens in document_tokens) / len(document_tokens) or 1.0
    lowered_phrases = tuple(phrase.casefold() for phrase in expanded_phrases)
    hits: list[SearchHit] = []
    k1, b = 1.2, 0.75

    for page, tokens in zip(pages, document_tokens, strict=True):
        frequencies = Counter(tokens)
        bm25 = 0.0
        for token, query_frequency in query_counts.items():
            term_frequency = frequencies[token]
            if term_frequency == 0:
                continue
            doc_frequency = document_frequencies[token]
            inverse_document_frequency = math.log(
                1 + (len(pages) - doc_frequency + 0.5) / (doc_frequency + 0.5)
            )
            denominator = term_frequency + k1 * (1 - b + b * len(tokens) / average_length)
            bm25 += (
                query_frequency
                * inverse_document_frequency
                * term_frequency
                * (k1 + 1)
                / denominator
            )

        lowered = page.text.lower()
        phrase_matches = sum(1 for phrase in lowered_phrases if phrase in lowered)
        exact_phrase = min(phrase_matches * 1.5, 4.5)
        year_signal = 1.25 if year is not None and str(year) in lowered else 0.0
        statement_signal = 4.0 if any(term in lowered for term in PRIMARY_STATEMENT_TERMS) else 0.0
        table_markers = lowered.count("|") + lowered.count("<td") + lowered.count("<tr")
        table_signal = min(table_markers / 20.0, 1.5)
        weak_penalty = -2.0 if any(term in lowered for term in WEAK_SECTION_TERMS) else 0.0
        preferred_statement = 20.0 if page.display_number in preferred_page_numbers else 0.0
        components = {
            "bm25": bm25,
            "exact_phrase": exact_phrase,
            "year": year_signal,
            "primary_statement": statement_signal,
            "table_density": table_signal,
            "weak_section_penalty": weak_penalty,
            "preferred_statement": preferred_statement,
        }
        score = sum(components.values())
        hits.append(SearchHit(page=page, score=score, components=components))

    return sorted(hits, key=lambda hit: (-hit.score, hit.page.raw_index))
