import os

import pytest

import config
from ingestion.pipeline import ingest_all
from rag.answer_engine import generate_answer
from rag.retriever import retrieve

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping live integration smoke test",
)


def test_end_to_end_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "GRAPH_PATH", tmp_path / "graph" / "kg.pkl")

    ingest_all(config.INCOMING_DIR)

    question = "Who performed the seal replacement on pump P-101?"
    retrieval = retrieve(question)
    result = generate_answer(question, retrieval)

    assert result.answer
    assert len(result.citations) >= 1
