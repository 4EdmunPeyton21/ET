# Industrial Knowledge Intelligence Copilot — Design Spec

**Date:** 2026-07-12
**Status:** Approved

## Context

This is the first sub-project under the broader "AI for Industrial Knowledge Intelligence"
hackathon challenge (see repo README / challenge brief). The full challenge scopes five
possible sub-systems: universal document ingestion + knowledge graph, an expert RAG copilot,
a maintenance/RCA agent, compliance intelligence, and a lessons-learned engine. Building all
five is out of scope for a hackathon prototype.

This spec covers the **foundational core**: a document ingestion pipeline that builds both a
vector index and a lightweight knowledge graph, plus a RAG-powered conversational copilot on
top of it. This is the piece most other sub-systems would depend on, and it directly
demonstrates the challenge's "time-to-answer vs. traditional search" and "knowledge graph
linkage completeness" evaluation criteria.

Maintenance/RCA, compliance intelligence, and lessons-learned are explicitly deferred to
future sub-projects, each to get its own spec/plan cycle if pursued.

## Goals

- Ingest heterogeneous industrial documents (PDFs, spreadsheets) into a searchable corpus
- Extract entities (equipment tags, dates, personnel, regulatory references, cross-document
  references) during ingestion
- Build a knowledge graph that links entities and documents together, updated incrementally
  as new documents arrive
- Provide a conversational copilot that answers questions across the corpus with source
  citations and a confidence score
- Run on a laptop with no external services beyond the Gemini API — no Docker, no hosted DB

## Non-Goals (deferred)

- OCR / scanned-image / P&ID computer-vision parsing
- Maintenance/RCA predictive analytics
- Regulatory compliance gap detection
- Cross-organization lessons-learned pattern mining
- Native mobile app (responsive web via Streamlit is sufficient for this phase)
- Multi-user auth, production deployment, horizontal scaling

## Architecture

```
[Synthetic Docs] → [Ingestion Pipeline] → [ChromaDB (vectors)] ─┐
                          │                                      ├→ [RAG Query Engine] → [Streamlit Chat UI]
                          └────────────→ [Knowledge Graph (NetworkX)] ─┘
```

Single Python project. Everything runs locally except Gemini API calls (embeddings +
generation). Ingestion is a batch step (run once, or re-run on new files); querying is
interactive via a Streamlit app.

## Components

- **`scripts/generate_synthetic_data.py`** — produces a mock plant document corpus: SOPs
  (PDF), maintenance work orders (PDF + CSV), inspection reports (PDF), equipment master list
  (CSV), regulatory reference sheet (PDF). Embeds realistic, consistent equipment tags (e.g.
  `P-101`, `HX-204`), dates, personnel names, and deliberate cross-references between
  documents so the knowledge graph has real relationships to discover.

- **`ingestion/parser.py`** — loads PDFs (pdfplumber) and spreadsheets (pandas/openpyxl) into
  raw text (PDFs) or structured rows (spreadsheets).

- **`ingestion/entity_extractor.py`** — calls the Gemini API with a structured-output schema
  per text chunk to extract entities: equipment tags, dates, personnel, regulatory
  references, and cross-document references.

- **`ingestion/graph_builder.py`** — builds/updates a NetworkX graph. Nodes: documents and
  entities. Edges: `mentions`, `performed_by`, `references`, `part_of`. Persisted to disk
  (pickle) so it survives restarts. Upserts are idempotent and dedup entities by a normalized
  key (e.g. uppercased equipment tag) so re-ingesting a document updates rather than
  duplicates.

- **`ingestion/embedder.py`** — chunks parsed text (~500 tokens, 50 overlap), gets Gemini
  embeddings per chunk, upserts into ChromaDB with metadata (source doc name, page number,
  entity tags found in that chunk).

- **`rag/retriever.py`** — given a user query: embeds it, runs top-k vector search in Chroma,
  looks up entities mentioned in the top hits within the knowledge graph, pulls in 1-hop
  neighbors (related docs/entities) as supplementary context, and assembles the final
  context bundle.

- **`rag/answer_engine.py`** — sends the assembled context + question to Gemini, requests an
  answer with inline citations (`[Source: doc_name, p.X]`) and a confidence label
  (High/Medium/Low — derived from top retrieval similarity score and whether the knowledge
  graph corroborates the answer with linked entities).

- **`app.py`** (Streamlit) — chat interface; renders the answer, a clickable source list, and
  a small graph visualization of the entities involved in the current answer. Streamlit's
  default responsive layout is used as the "mobile" story for field technicians.

## Data Flow

**Ingestion (batch):**
1. Files placed in `data/incoming/`
2. Parser extracts raw text (PDF) or rows (spreadsheet) per file
3. Text chunked (~500 tokens, 50 overlap)
4. Entity extractor runs per chunk → structured entities
5. Graph builder upserts nodes/edges (idempotent)
6. Embedder writes chunk vectors + metadata to Chroma

**Query (interactive):**
1. User asks a question in the Streamlit chat
2. Query embedded → top-k Chroma similarity search (optional metadata filter by doc type)
3. Entities from top hits looked up in the graph → 1-hop neighbors pulled as extra context
4. Context (chunks + graph neighborhood) assembled into the generation prompt
5. Gemini generates an answer with citations and a confidence label
6. Streamlit renders the answer, source list, and entity graph snippet

## Error Handling

- Malformed/unparseable file → log and skip, surfaced in an ingestion summary report rather
  than crashing the batch
- Gemini API failure (rate limit/timeout) → retry with backoff (3 attempts); if still
  failing, the chunk is marked "pending" for re-run during ingestion, and at query time the
  answer engine falls back to a plain "couldn't generate an answer right now" message rather
  than surfacing an exception
- No relevant retrieval hits / all scores below similarity threshold → copilot responds "No
  matching information found in the corpus" instead of hallucinating an answer
- Duplicate/redundant entity extraction across chunks → deduped at the graph-builder level by
  normalized entity key

## Testing

- Unit tests for `parser.py` (given a sample PDF/CSV, correct text/rows are extracted)
- Unit tests for `graph_builder.py` (idempotent upsert behavior, dedup logic)
- A benchmark Q&A set (~10-15 questions with known answers derivable from the synthetic
  corpus), used to check answer quality and citation correctness — maps directly to the
  challenge's "query answer quality on domain-expert benchmark questions" criterion
- Smoke test: run full ingestion end-to-end, ask one benchmark question, assert an answer
  with at least one citation is returned

## Tech Stack

- Python, FastAPI-free (Streamlit serves as both backend and frontend for this phase)
- Parsing: pdfplumber, pandas/openpyxl
- Embeddings + generation: Gemini API
- Vector store: ChromaDB (embedded/local)
- Knowledge graph: NetworkX (persisted via pickle)
- UI: Streamlit
