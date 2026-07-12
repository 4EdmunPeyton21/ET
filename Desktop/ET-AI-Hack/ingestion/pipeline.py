from pathlib import Path

import config
from ingestion.embedder import ChunkRecord, chunk_text, get_collection, upsert_chunk
from ingestion.entity_extractor import extract_entities
from ingestion.graph_builder import GraphBuilder
from ingestion.parser import parse_document


def ingest_all(incoming_dir: Path = config.INCOMING_DIR) -> dict:
    graph_builder = GraphBuilder(config.GRAPH_PATH)
    collection = get_collection()
    report = {"ingested": [], "skipped": []}

    for path in sorted(incoming_dir.glob("*")):
        if path.is_dir():
            continue
        try:
            doc_type, pages = parse_document(path)
        except Exception as exc:
            report["skipped"].append({"file": path.name, "error": str(exc)})
            continue

        doc_id = path.name
        graph_builder.upsert_document(doc_id, doc_type)

        for page in pages:
            for i, chunk in enumerate(chunk_text(page.text)):
                chunk_id = f"{doc_id}::p{page.page_number}::c{i}"
                try:
                    graph_builder.upsert_chunk(chunk_id, doc_id, page.page_number, chunk)
                    entities = extract_entities(chunk)
                    graph_builder.upsert_entities(chunk_id, entities)
                    upsert_chunk(collection, ChunkRecord(chunk_id, doc_id, doc_type, page.page_number, chunk))
                except Exception as exc:
                    report["skipped"].append({"file": chunk_id, "error": str(exc)})
                    continue

        report["ingested"].append(doc_id)
        graph_builder.save()

    graph_builder.save()
    return report


if __name__ == "__main__":
    result = ingest_all()
    print(f"Ingested: {result['ingested']}")
    if result["skipped"]:
        print(f"Skipped (errors): {result['skipped']}")
