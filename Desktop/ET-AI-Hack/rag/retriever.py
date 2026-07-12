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
class GraphContextChunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str


@dataclass
class RetrievalResult:
    chunks: list[ContextChunk]
    graph_entities: list[str]
    graph_context_chunks: list[GraphContextChunk]


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

    # Genuine 2-hop traversal: entity -> other chunks that mention it -> those
    # chunks' text/document/entities. This is what actually lets the graph
    # surface cross-document context (e.g. an equipment tag mentioned in both
    # an SOP and a work order) instead of being a structural no-op.
    direct_chunk_ids = {chunk.chunk_id for chunk in chunks}
    related_chunk_ids = set()
    for entity_id in entity_ids:
        if entity_id not in graph_builder.graph:
            continue
        for pred_id in graph_builder.graph.predecessors(entity_id):
            if pred_id in direct_chunk_ids:
                continue
            if graph_builder.graph.nodes[pred_id].get("type") != "chunk":
                continue
            related_chunk_ids.add(pred_id)

    graph_context_chunks = []
    for chunk_id in related_chunk_ids:
        node_data = graph_builder.graph.nodes[chunk_id]
        doc_id = None
        for succ_id in graph_builder.graph.successors(chunk_id):
            if graph_builder.graph.nodes[succ_id].get("type") == "document":
                doc_id = succ_id
                break
        if doc_id is None:
            continue
        graph_context_chunks.append(
            GraphContextChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page=node_data.get("page"),
                text=node_data.get("text"),
            )
        )

    # Cap fan-out: a common entity shared by many chunks (large real corpora,
    # unlike this small demo one) could otherwise pull an unbounded amount of
    # text into the prompt. Sort for deterministic selection.
    graph_context_chunks.sort(key=lambda c: c.chunk_id)
    graph_context_chunks = graph_context_chunks[: config.MAX_GRAPH_CONTEXT_CHUNKS]

    related_entity_ids = set()
    for chunk in graph_context_chunks:
        related_entity_ids.update(graph_builder.entities_for_chunk(chunk.chunk_id))

    graph_entities = sorted(entity_ids | filtered_neighbor_entities | related_entity_ids)

    return RetrievalResult(
        chunks=chunks,
        graph_entities=graph_entities,
        graph_context_chunks=graph_context_chunks,
    )
