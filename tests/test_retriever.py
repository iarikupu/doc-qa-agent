import pytest

from docqa.retriever import Chunk, Retriever, split_into_chunks


def test_split_drops_tiny_fragments_and_indexes_chunks():
    text = "# Title\n\nThis paragraph is comfortably long enough to be kept as a chunk.\n\nshort\n\nAnother paragraph that is also long enough to be a real chunk."
    chunks = split_into_chunks("doc.md", text)
    assert [c.index for c in chunks] == [0, 1]
    assert all(c.source == "doc.md" for c in chunks)


def test_search_ranks_the_relevant_chunk_first():
    retriever = Retriever(
        [
            Chunk("a.md", 0, "Cats are small domesticated carnivorous mammals kept as pets."),
            Chunk("b.md", 0, "Python is a programming language used for data and web development."),
        ]
    )
    results = retriever.search("which programming language is used for web development")
    assert results[0][0].source == "b.md"
    assert results[0][1] > 0


def test_search_returns_nothing_for_unrelated_query():
    retriever = Retriever([Chunk("a.md", 0, "Cats are small domesticated carnivorous mammals.")])
    assert retriever.search("quantum chromodynamics lagrangian") == []


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        Retriever([])


def test_from_directory_loads_markdown_files(tmp_path):
    (tmp_path / "x.md").write_text("A sufficiently long paragraph about expense reports and receipts.", encoding="utf-8")
    retriever = Retriever.from_directory(tmp_path)
    assert retriever.chunks[0].source == "x.md"
