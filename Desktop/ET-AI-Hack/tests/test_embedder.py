from ingestion.embedder import chunk_text


def test_chunk_text_splits_on_word_count():
    text = " ".join(f"word{i}" for i in range(120))
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 3
    assert chunks[0].split()[0] == "word0"


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("", chunk_size=50, overlap=10) == []


def test_chunk_text_short_text_returns_single_chunk():
    chunks = chunk_text("hello world", chunk_size=50, overlap=10)
    assert chunks == ["hello world"]
