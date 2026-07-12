from dataclasses import dataclass

import config
from ingestion.embedder import embed_text, get_collection
from ingestion.graph_builder import GraphBuilder


@dataclass
class ContextChunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    similarity: float


@dataclass
class RetrievalResult:
    chunks: list[ContextChunk]
    graph_entities: list[str]


def retrieve(question: str, top_k: int = config.TOP_K) -> RetrievalResult:
    collection = get_collection()
    query_embedding = embed_text(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = [
        ContextChunk(
            chunk_id=chunk_id,
            doc_id=metadata["doc_id"],
            page=metadata["page"],
            text=text,
            similarity=1 - distance,
        )
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
    ]

    graph_builder = GraphBuilder(config.GRAPH_PATH)
    entity_ids = set()
    for chunk in chunks:
        entity_ids.update(graph_builder.entities_for_chunk(chunk.chunk_id))
    neighbor_entities = graph_builder.get_neighbors(list(entity_ids), hops=1)
    filtered_neighbor_entities = {
        node_id
        for node_id in neighbor_entities
        if node_id in graph_builder.graph
        and graph_builder.graph.nodes[node_id].get("type") not in ("chunk", "document")
    }

    return RetrievalResult(chunks=chunks, graph_entities=sorted(entity_ids | filtered_neighbor_entities))
