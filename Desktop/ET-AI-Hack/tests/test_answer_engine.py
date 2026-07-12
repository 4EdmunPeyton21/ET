from rag.answer_engine import _confidence_label


def test_confidence_high_when_similarity_and_graph_corroborate():
    assert _confidence_label(0.8, True) == "High"


def test_confidence_medium_when_similarity_moderate():
    assert _confidence_label(0.6, False) == "Medium"


def test_confidence_low_when_similarity_weak():
    assert _confidence_label(0.2, False) == "Low"
