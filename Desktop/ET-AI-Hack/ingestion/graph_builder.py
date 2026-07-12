import pickle
from pathlib import Path

import networkx as nx

ENTITY_EDGE_TYPES = {
    "equipment_tags": "mentions",
    "dates": "mentions",
    "regulatory_refs": "mentions",
    "document_refs": "references",
    "personnel": "performed_by",
}


class GraphBuilder:
    def __init__(self, graph_path: Path):
        self.graph_path = graph_path
        if graph_path.exists():
            with open(graph_path, "rb") as f:
                self.graph = pickle.load(f)
        else:
            self.graph = nx.MultiDiGraph()

    def upsert_document(self, doc_id: str, doc_type: str) -> None:
        self.graph.add_node(doc_id, type="document", doc_type=doc_type)

    def upsert_chunk(self, chunk_id: str, doc_id: str, page: int, text: str) -> None:
        self.graph.add_node(chunk_id, type="chunk", page=page, text=text)
        if not self.graph.has_edge(chunk_id, doc_id, key="part_of"):
            self.graph.add_edge(chunk_id, doc_id, key="part_of", relation="part_of")

    @staticmethod
    def normalize_entity_id(entity_type: str, value: str) -> str:
        return f"{entity_type}:{value.strip().upper()}"

    def upsert_entities(self, chunk_id: str, entities: dict) -> None:
        for entity_type, values in entities.items():
            edge_relation = ENTITY_EDGE_TYPES.get(entity_type, "mentions")
            for value in values:
                if not value:
                    continue
                entity_id = self.normalize_entity_id(entity_type, value)
                self.graph.add_node(entity_id, type=entity_type, value=value)
                if not self.graph.has_edge(chunk_id, entity_id, key=edge_relation):
                    self.graph.add_edge(chunk_id, entity_id, key=edge_relation, relation=edge_relation)

    def entities_for_chunk(self, chunk_id: str) -> list[str]:
        if chunk_id not in self.graph:
            return []
        return [
            n for n in self.graph.successors(chunk_id)
            if self.graph.nodes[n].get("type") != "document"
        ]

    def get_neighbors(self, node_ids: list[str], hops: int = 1) -> set[str]:
        neighbors = set()
        frontier = set(node_ids)
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                if node not in self.graph:
                    continue
                next_frontier.update(self.graph.successors(node))
                next_frontier.update(self.graph.predecessors(node))
            neighbors.update(next_frontier)
            frontier = next_frontier
        return neighbors - set(node_ids)

    def save(self) -> None:
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.graph, f)
