from dataclasses import dataclass

import chromadb
import google.generativeai as genai

import config

genai.configure(api_key=config.GEMINI_API_KEY)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    doc_type: str
    page: int
    text: str


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE_WORDS,
    overlap: int = config.CHUNK_OVERLAP_WORDS,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def embed_text(text: str) -> list[float]:
    result = genai.embed_content(model=config.GEMINI_EMBEDDING_MODEL, content=text)
    return result["embedding"]


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(name="industrial_docs", metadata={"hnsw:space": "cosine"})


def upsert_chunk(collection, record: ChunkRecord) -> None:
    embedding = embed_text(record.text)
    collection.upsert(
        ids=[record.chunk_id],
        embeddings=[embedding],
        documents=[record.text],
        metadatas=[{"doc_id": record.doc_id, "doc_type": record.doc_type, "page": record.page}],
    )
