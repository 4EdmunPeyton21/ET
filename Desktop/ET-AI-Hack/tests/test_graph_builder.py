# tests/test_graph_builder.py
from ingestion.graph_builder import GraphBuilder


def test_upsert_document_is_idempotent(tmp_path):
    gb = GraphBuilder(tmp_path / "graph.pkl")
    gb.upsert_document("doc1.pdf", "pdf")
    gb.upsert_document("doc1.pdf", "pdf")
    assert gb.graph.number_of_nodes() == 1


def test_upsert_entities_dedup_by_normalized_key(tmp_path):
    gb = GraphBuilder(tmp_path / "graph.pkl")
    gb.upsert_document("doc1.pdf", "pdf")
    gb.upsert_chunk("doc1.pdf::chunk0", "doc1.pdf", 1, "text")
    gb.upsert_entities("doc1.pdf::chunk0", {"equipment_tags": ["p-101", "P-101"]})

    equipment_nodes = [n for n, d in gb.graph.nodes(data=True) if d.get("type") == "equipment_tags"]
    assert equipment_nodes == ["equipment_tags:P-101"]


def test_save_and_reload_persists_graph(tmp_path):
    graph_path = tmp_path / "graph.pkl"
    gb = GraphBuilder(graph_path)
    gb.upsert_document("doc1.pdf", "pdf")
    gb.save()

    gb2 = GraphBuilder(graph_path)
    assert "doc1.pdf" in gb2.graph.nodes


def test_get_neighbors_one_hop(tmp_path):
    gb = GraphBuilder(tmp_path / "graph.pkl")
    gb.upsert_document("doc1.pdf", "pdf")
    gb.upsert_chunk("doc1.pdf::chunk0", "doc1.pdf", 1, "text")
    gb.upsert_entities("doc1.pdf::chunk0", {"equipment_tags": ["P-101"]})

    neighbors = gb.get_neighbors(["equipment_tags:P-101"], hops=1)
    assert "doc1.pdf::chunk0" in neighbors


def test_entities_for_chunk_excludes_document_node(tmp_path):
    gb = GraphBuilder(tmp_path / "graph.pkl")
    gb.upsert_document("doc1.pdf", "pdf")
    gb.upsert_chunk("doc1.pdf::chunk0", "doc1.pdf", 1, "text")
    gb.upsert_entities("doc1.pdf::chunk0", {"personnel": ["Rajesh Kumar"]})

    entities = gb.entities_for_chunk("doc1.pdf::chunk0")
    assert entities == ["personnel:RAJESH KUMAR"]
