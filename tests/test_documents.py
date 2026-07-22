from finground.documents import split_pages


def test_split_pages_preserves_raw_split_positions() -> None:
    pages = split_pages("first<--- Page Split --->   <--- Page Split --->third")

    assert [page.raw_index for page in pages] == [0, 2]
    assert [page.display_number for page in pages] == [1, 3]
