# Industrial Knowledge Copilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a document ingestion pipeline (PDFs + spreadsheets → entity extraction → knowledge graph + vector store) and a RAG-powered Streamlit copilot that answers questions with source citations and a confidence score.

**Architecture:** Synthetic industrial documents are ingested through a pipeline that parses text, extracts entities via Gemini, builds a NetworkX knowledge graph, and embeds chunks into ChromaDB. A Streamlit chat app retrieves relevant chunks (vector search + 1-hop graph expansion) and asks Gemini to generate a cited answer.

**Tech Stack:** Python, pdfplumber, pandas/openpyxl, fpdf2, google-generativeai (Gemini), ChromaDB, NetworkX, Streamlit, pytest.

## Global Constraints

- Python only, single local process — no Docker, no hosted DB, no external services besides the Gemini API (per spec Architecture/Tech Stack)
- Document types: PDF and spreadsheet (CSV/XLSX) only — no OCR, no scanned images, no P&ID computer vision (per spec Non-Goals)
- No auth, no multi-user support, no production deployment concerns (per spec Non-Goals)
- Vector store: ChromaDB, embedded/local persistent client (per spec Tech Stack)
- Knowledge graph: NetworkX, persisted to disk via pickle (per spec Tech Stack)
- UI: Streamlit only (per spec Components / UI)
- Gemini model IDs are read from environment variables with defaults (`gemini-2.0-flash`, `models/text-embedding-004`) — **verify these are still valid, non-deprecated model IDs against the current Gemini API docs before running**, since model availability changes over time and this plan was written with a Feb-2025 knowledge cutoff
- No hallucinated answers: if retrieval similarity is below threshold, the copilot must say so explicitly rather than guessing (per spec Error Handling)

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `config.py`
- Create: `ingestion/__init__.py`
- Create: `rag/__init__.py`

**Interfaces:**
- Produces: `config.GEMINI_API_KEY`, `config.GEMINI_MODEL`, `config.GEMINI_EMBEDDING_MODEL`, `config.DATA_DIR`, `config.INCOMING_DIR`, `config.CHROMA_DIR`, `config.GRAPH_PATH`, `config.CHUNK_SIZE_WORDS` (int), `config.CHUNK_OVERLAP_WORDS` (int), `config.TOP_K` (int), `config.SIMILARITY_THRESHOLD` (float) — all consumed by every later task.

- [ ] **Step 1: Create `requirements.txt`**

```
google-generativeai>=0.8.0
chromadb>=0.5.0
networkx>=3.0
pdfplumber>=0.11.0
fpdf2>=2.7.0
pandas>=2.0.0
openpyxl>=3.1.0
streamlit>=1.35.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.env
data/
```

- [ ] **Step 3: Create `.env.example`**

```
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
```

- [ ] **Step 4: Create `config.py`**

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INCOMING_DIR = DATA_DIR / "incoming"
CHROMA_DIR = DATA_DIR / "chroma"
GRAPH_PATH = DATA_DIR / "graph" / "knowledge_graph.pkl"

CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
```

- [ ] **Step 5: Create empty package markers**

```bash
touch ingestion/__init__.py rag/__init__.py
```

- [ ] **Step 6: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: all packages install without error

- [ ] **Step 7: Verify config loads**

Run: `python -c "import config; print(config.GEMINI_MODEL, config.CHUNK_SIZE_WORDS)"`
Expected: `gemini-2.0-flash 500`

- [ ] **Step 8: Commit**

```bash
git add requirements.txt .gitignore .env.example config.py ingestion/__init__.py rag/__init__.py
git commit -m "Scaffold project: config, dependencies, package structure"
```

---

### Task 2: Synthetic Data Generator

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/generate_synthetic_data.py`

**Interfaces:**
- Consumes: `config.INCOMING_DIR`
- Produces: `generate_all(out_dir: Path = config.INCOMING_DIR) -> None` — writes 5 files (2 PDFs analyzed below + 1 more PDF + 2 CSVs) into `out_dir`. Later tasks (pipeline) read these files by scanning `config.INCOMING_DIR`.

- [ ] **Step 1: Create `scripts/__init__.py`**

```bash
touch scripts/__init__.py
```

- [ ] **Step 2: Write `scripts/generate_synthetic_data.py`**

```python
import csv
from pathlib import Path
from fpdf import FPDF

import config


def write_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, title)
    pdf.set_font("Helvetica", "", 12)
    pdf.ln(4)
    for para in paragraphs:
        pdf.multi_cell(0, 8, para)
        pdf.ln(2)
    pdf.output(str(path))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_sop_pump_maintenance(out_dir: Path) -> None:
    write_pdf(
        out_dir / "sop_pump_maintenance.pdf",
        "SOP-014: Preventive Maintenance for Feed Pump P-101",
        [
            "Document ID: SOP-014",
            "Equipment: P-101 (Feed Pump)",
            "Approved by: Rajesh Kumar",
            "Regulatory Reference: OISD-STD-118",
            "This procedure covers routine preventive maintenance of pump P-101, "
            "including seal inspection and lubrication checks, to be performed quarterly.",
            "Related work order: WO-2044 documents the most recent seal replacement "
            "performed on P-101.",
            "Any deviation from this procedure must be logged and reported per "
            "OISD-STD-118 section 4.2.",
        ],
    )


def generate_inspection_report_hx204(out_dir: Path) -> None:
    write_pdf(
        out_dir / "inspection_report_hx204.pdf",
        "Inspection Report: Heat Exchanger HX-204",
        [
            "Document ID: INSP-2026-031",
            "Equipment: HX-204 (Shell and Tube Heat Exchanger)",
            "Inspection Date: 2026-03-25",
            "Inspector: Anita Sharma",
            "Findings: Tube bundle fouling within acceptable limits after cleaning. "
            "This inspection follows the cleaning work performed under WO-2045.",
            "Regulatory Reference: PESO-CCE-2019",
            "Recommendation: Re-inspect in 6 months or after any process upset.",
        ],
    )


def generate_regulatory_reference_sheet(out_dir: Path) -> None:
    write_pdf(
        out_dir / "regulatory_reference_sheet.pdf",
        "Regulatory Reference Sheet",
        [
            "OISD-STD-118: Covers preventive maintenance practices for rotating "
            "equipment including pumps and compressors in the process industry.",
            "PESO-CCE-2019: Petroleum and Explosives Safety Organisation guidelines "
            "for pressure vessel and heat exchanger inspection intervals.",
            "Factory-Act-1948-Sec-87: Statutory requirement for reporting dangerous "
            "occurrences involving pressure plant, including vessels such as V-301.",
        ],
    )


def generate_work_orders_csv(out_dir: Path) -> None:
    rows = [
        {"work_order_id": "WO-2044", "equipment_tag": "P-101", "date": "2026-03-14",
         "performed_by": "Rajesh Kumar", "description": "Replaced pump mechanical seal"},
        {"work_order_id": "WO-2045", "equipment_tag": "HX-204", "date": "2026-03-20",
         "performed_by": "Anita Sharma", "description": "Cleaned tube bundle, removed fouling"},
        {"work_order_id": "WO-2046", "equipment_tag": "V-301", "date": "2026-04-02",
         "performed_by": "Mohammed Iqbal", "description": "Inspected pressure relief valve"},
    ]
    _write_csv(out_dir / "work_orders.csv", rows)


def generate_equipment_master_csv(out_dir: Path) -> None:
    rows = [
        {"equipment_tag": "P-101", "description": "Feed Pump", "location": "Unit 1", "install_date": "2018-06-01"},
        {"equipment_tag": "HX-204", "description": "Shell and Tube Heat Exchanger", "location": "Unit 2", "install_date": "2015-11-12"},
        {"equipment_tag": "V-301", "description": "Separator Vessel", "location": "Unit 3", "install_date": "2012-02-20"},
        {"equipment_tag": "C-102", "description": "Feed Gas Compressor", "location": "Unit 1", "install_date": "2019-09-05"},
    ]
    _write_csv(out_dir / "equipment_master.csv", rows)


def generate_all(out_dir: Path = config.INCOMING_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_sop_pump_maintenance(out_dir)
    generate_inspection_report_hx204(out_dir)
    generate_regulatory_reference_sheet(out_dir)
    generate_work_orders_csv(out_dir)
    generate_equipment_master_csv(out_dir)
    print(f"Generated synthetic corpus in {out_dir}")


if __name__ == "__main__":
    generate_all()
```

- [ ] **Step 3: Run it and verify output**

Run: `python -m scripts.generate_synthetic_data`
Expected: prints `Generated synthetic corpus in .../data/incoming`, and `data/incoming/` contains 3 PDFs + 2 CSVs

- [ ] **Step 4: Commit**

```bash
git add scripts/__init__.py scripts/generate_synthetic_data.py
git commit -m "Add synthetic industrial document generator"
```

---

### Task 3: Document Parser

**Files:**
- Create: `ingestion/parser.py`
- Test: `tests/test_parser.py`

**Interfaces:**
- Consumes: nothing beyond stdlib/pdfplumber/pandas
- Produces: `ParsedPage(page_number: int, text: str)` dataclass; `parse_document(path: Path) -> tuple[str, list[ParsedPage]]` where the returned `str` is `"pdf"` or `"spreadsheet"`. Consumed by `ingestion/pipeline.py` (Task 7).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parser.py
from pathlib import Path
from fpdf import FPDF

from ingestion.parser import parse_document, doc_type_for


def _make_pdf(path: Path, lines: list[str]) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 12)
    for line in lines:
        pdf.multi_cell(0, 8, line)
    pdf.output(str(path))


def test_doc_type_for_pdf_and_spreadsheet():
    assert doc_type_for(Path("a.pdf")) == "pdf"
    assert doc_type_for(Path("a.csv")) == "spreadsheet"
    assert doc_type_for(Path("a.xlsx")) == "spreadsheet"


def test_parse_pdf_extracts_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, ["Equipment P-101 preventive maintenance."])

    doc_type, pages = parse_document(pdf_path)

    assert doc_type == "pdf"
    assert len(pages) == 1
    assert "P-101" in pages[0].text
    assert pages[0].page_number == 1


def test_parse_csv_extracts_rows(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("equipment_tag,description\nP-101,Feed Pump\n")

    doc_type, pages = parse_document(csv_path)

    assert doc_type == "spreadsheet"
    assert len(pages) == 1
    assert "P-101" in pages[0].text
    assert "Feed Pump" in pages[0].text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.parser'`

- [ ] **Step 3: Write `ingestion/parser.py`**

```python
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pdfplumber


@dataclass
class ParsedPage:
    page_number: int
    text: str


def doc_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in (".csv", ".xlsx", ".xls"):
        return "spreadsheet"
    raise ValueError(f"Unsupported file type: {suffix}")


def parse_pdf(path: Path) -> list[ParsedPage]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(ParsedPage(page_number=i, text=text))
    return pages


def parse_spreadsheet(path: Path) -> list[ParsedPage]:
    if path.suffix.lower() == ".csv":
        sheets = {path.stem: pd.read_csv(path)}
    else:
        sheets = pd.read_excel(path, sheet_name=None)

    pages = []
    for i, (sheet_name, df) in enumerate(sheets.items(), start=1):
        lines = [f"Sheet: {sheet_name}", ", ".join(str(c) for c in df.columns)]
        for _, row in df.iterrows():
            lines.append(", ".join(str(v) for v in row.values))
        pages.append(ParsedPage(page_number=i, text="\n".join(lines)))
    return pages


def parse_document(path: Path) -> tuple[str, list[ParsedPage]]:
    doc_type = doc_type_for(path)
    if doc_type == "pdf":
        return doc_type, parse_pdf(path)
    return doc_type, parse_spreadsheet(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parser.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/parser.py tests/test_parser.py
git commit -m "Add document parser for PDFs and spreadsheets"
```

---

### Task 4: Entity Extractor

**Files:**
- Create: `ingestion/entity_extractor.py`
- Test: `tests/test_entity_extractor.py`

**Interfaces:**
- Consumes: `config.GEMINI_API_KEY`, `config.GEMINI_MODEL`
- Produces: `ENTITY_KEYS: list[str]` (`["equipment_tags", "dates", "personnel", "regulatory_refs", "document_refs"]`); `extract_entities(text: str, max_retries: int = 3) -> dict[str, list[str]]`. Consumed by `ingestion/pipeline.py` (Task 7) and `ingestion/graph_builder.py`'s `upsert_entities` (Task 5, which takes this dict shape as input).

- [ ] **Step 1: Write the failing test (pure parsing logic only — no live API calls)**

```python
# tests/test_entity_extractor.py
from ingestion.entity_extractor import _parse_response, ENTITY_KEYS


def test_parse_response_valid_json():
    raw = '{"equipment_tags": ["P-101"], "dates": ["2026-03-14"], "personnel": [], "regulatory_refs": [], "document_refs": []}'
    result = _parse_response(raw)
    assert result["equipment_tags"] == ["P-101"]
    assert result["dates"] == ["2026-03-14"]
    assert set(result.keys()) == set(ENTITY_KEYS)


def test_parse_response_strips_markdown_fences():
    raw = '```json\n{"equipment_tags": ["HX-204"], "dates": [], "personnel": [], "regulatory_refs": [], "document_refs": []}\n```'
    result = _parse_response(raw)
    assert result["equipment_tags"] == ["HX-204"]


def test_parse_response_invalid_json_returns_empty_lists():
    result = _parse_response("not json at all")
    assert all(result[key] == [] for key in ENTITY_KEYS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_entity_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.entity_extractor'`

- [ ] **Step 3: Write `ingestion/entity_extractor.py`**

```python
import json
import time

import google.generativeai as genai

import config

genai.configure(api_key=config.GEMINI_API_KEY)

ENTITY_KEYS = ["equipment_tags", "dates", "personnel", "regulatory_refs", "document_refs"]

PROMPT_TEMPLATE = """You are an industrial document entity extractor. Extract entities from the text below.

Return ONLY valid JSON (no markdown fences, no commentary) matching this exact shape:
{{
  "equipment_tags": ["<equipment tag strings, e.g. P-101, HX-204>"],
  "dates": ["<dates as they appear in the text>"],
  "personnel": ["<person names>"],
  "regulatory_refs": ["<regulatory/standard references, e.g. OISD-STD-118>"],
  "document_refs": ["<references to other document IDs, e.g. WO-2044>"]
}}

If a category has no entities, return an empty list for it.

Text:
\"\"\"
{text}
\"\"\"
"""


def _parse_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {key: [] for key in ENTITY_KEYS}
    return {key: [str(v) for v in data.get(key, [])] for key in ENTITY_KEYS}


def extract_entities(text: str, max_retries: int = 3) -> dict:
    if not text.strip():
        return {key: [] for key in ENTITY_KEYS}

    model = genai.GenerativeModel(config.GEMINI_MODEL)
    prompt = PROMPT_TEMPLATE.format(text=text[:6000])

    last_error = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return _parse_response(response.text)
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)

    print(f"entity_extractor: giving up after {max_retries} attempts: {last_error}")
    return {key: [] for key in ENTITY_KEYS}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_entity_extractor.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/entity_extractor.py tests/test_entity_extractor.py
git commit -m "Add Gemini-based entity extractor with JSON parsing and retry"
```

---

### Task 5: Knowledge Graph Builder

**Files:**
- Create: `ingestion/graph_builder.py`
- Test: `tests/test_graph_builder.py`

**Interfaces:**
- Consumes: entity dict shape from Task 4 (`{"equipment_tags": [...], "dates": [...], "personnel": [...], "regulatory_refs": [...], "document_refs": [...]}`)
- Produces: `GraphBuilder(graph_path: Path)` with methods `upsert_document(doc_id: str, doc_type: str) -> None`, `upsert_chunk(chunk_id: str, doc_id: str, page: int, text: str) -> None`, `upsert_entities(chunk_id: str, entities: dict) -> None`, `entities_for_chunk(chunk_id: str) -> list[str]`, `get_neighbors(node_ids: list[str], hops: int = 1) -> set[str]`, `save() -> None`. Consumed by `ingestion/pipeline.py` (Task 7) and `rag/retriever.py` (Task 8).

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.graph_builder'`

- [ ] **Step 3: Write `ingestion/graph_builder.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_builder.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/graph_builder.py tests/test_graph_builder.py
git commit -m "Add knowledge graph builder with idempotent upserts and neighbor lookup"
```

---

### Task 6: Chunking, Embeddings, and Vector Store

**Files:**
- Create: `ingestion/embedder.py`

**Interfaces:**
- Consumes: `config.CHUNK_SIZE_WORDS`, `config.CHUNK_OVERLAP_WORDS`, `config.CHROMA_DIR`, `config.GEMINI_EMBEDDING_MODEL`, `config.GEMINI_API_KEY`
- Produces: `ChunkRecord(chunk_id, doc_id, doc_type, page, text)` dataclass; `chunk_text(text: str, chunk_size: int = ..., overlap: int = ...) -> list[str]`; `embed_text(text: str) -> list[float]`; `get_collection()` (returns a Chroma collection handle); `upsert_chunk(collection, record: ChunkRecord) -> None`. Consumed by `ingestion/pipeline.py` (Task 7) and `rag/retriever.py` (Task 8, which calls `embed_text` and `get_collection`).

- [ ] **Step 1: Write the failing test (pure chunking logic only — no API calls)**

```python
# tests/test_embedder.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.embedder'`

- [ ] **Step 3: Write `ingestion/embedder.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embedder.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/embedder.py tests/test_embedder.py
git commit -m "Add chunking, Gemini embeddings, and ChromaDB upsert"
```

---

### Task 7: Ingestion Pipeline Orchestrator

**Files:**
- Create: `ingestion/pipeline.py`

**Interfaces:**
- Consumes: `config.INCOMING_DIR`, `config.GRAPH_PATH`; `parse_document` (Task 3); `extract_entities` (Task 4); `GraphBuilder` (Task 5); `chunk_text`, `get_collection`, `upsert_chunk`, `ChunkRecord` (Task 6)
- Produces: `ingest_all(incoming_dir: Path = config.INCOMING_DIR) -> dict` returning `{"ingested": [doc_id, ...], "skipped": [{"file": ..., "error": ...}, ...]}`. Consumed by `tests/test_smoke.py` (Task 11) and run manually for the demo.

- [ ] **Step 1: Write `ingestion/pipeline.py`**

```python
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
                graph_builder.upsert_chunk(chunk_id, doc_id, page.page_number, chunk)
                entities = extract_entities(chunk)
                graph_builder.upsert_entities(chunk_id, entities)
                upsert_chunk(collection, ChunkRecord(chunk_id, doc_id, doc_type, page.page_number, chunk))

        report["ingested"].append(doc_id)

    graph_builder.save()
    return report


if __name__ == "__main__":
    result = ingest_all()
    print(f"Ingested: {result['ingested']}")
    if result["skipped"]:
        print(f"Skipped (errors): {result['skipped']}")
```

- [ ] **Step 2: Run the pipeline against the synthetic corpus**

Run: `python -m ingestion.pipeline`
Expected: prints `Ingested: ['equipment_master.csv', 'inspection_report_hx204.pdf', 'regulatory_reference_sheet.pdf', 'sop_pump_maintenance.pdf', 'work_orders.csv']` with no skipped entries (requires `GEMINI_API_KEY` set in `.env`)

- [ ] **Step 3: Verify the graph and vector store were created**

Run: `python -c "from pathlib import Path; import config; print(config.GRAPH_PATH.exists(), any(config.CHROMA_DIR.glob('*')))"`
Expected: `True True`

- [ ] **Step 4: Commit**

```bash
git add ingestion/pipeline.py
git commit -m "Add ingestion pipeline orchestrator"
```

---

### Task 8: Retriever

**Files:**
- Create: `rag/retriever.py`

**Interfaces:**
- Consumes: `config.TOP_K`, `config.GRAPH_PATH`; `embed_text`, `get_collection` (Task 6); `GraphBuilder` (Task 5)
- Produces: `ContextChunk(chunk_id, doc_id, page, text, similarity)` dataclass; `RetrievalResult(chunks: list[ContextChunk], graph_entities: list[str])` dataclass; `retrieve(question: str, top_k: int = config.TOP_K) -> RetrievalResult`. Consumed by `rag/answer_engine.py` (Task 9) and `app.py` (Task 10).

- [ ] **Step 1: Write `rag/retriever.py`**

```python
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

    return RetrievalResult(chunks=chunks, graph_entities=sorted(entity_ids | neighbor_entities))
```

- [ ] **Step 2: Manually verify retrieval against the ingested corpus**

Run: `python -c "from rag.retriever import retrieve; r = retrieve('Who performed maintenance on P-101?'); print([c.doc_id for c in r.chunks]); print(r.graph_entities)"`
Expected: `sop_pump_maintenance.pdf` and/or `work_orders.csv` appear in the doc_id list; `graph_entities` includes entries like `personnel:RAJESH KUMAR` and `equipment_tags:P-101`

- [ ] **Step 3: Commit**

```bash
git add rag/retriever.py
git commit -m "Add retriever combining vector search with graph neighbor expansion"
```

---

### Task 9: Answer Engine

**Files:**
- Create: `rag/answer_engine.py`

**Interfaces:**
- Consumes: `config.GEMINI_API_KEY`, `config.GEMINI_MODEL`, `config.SIMILARITY_THRESHOLD`; `RetrievalResult`, `ContextChunk` (Task 8)
- Produces: `Citation(doc_id, page)` dataclass; `AnswerResult(answer: str, citations: list[Citation], confidence: str)` dataclass; `generate_answer(question: str, retrieval: RetrievalResult) -> AnswerResult`. Consumed by `app.py` (Task 10) and `benchmark/run_benchmark.py` (Task 11).

- [ ] **Step 1: Write the failing test (pure confidence-labeling logic only — no API calls)**

```python
# tests/test_answer_engine.py
from rag.answer_engine import _confidence_label


def test_confidence_high_when_similarity_and_graph_corroborate():
    assert _confidence_label(0.8, True) == "High"


def test_confidence_medium_when_similarity_moderate():
    assert _confidence_label(0.6, False) == "Medium"


def test_confidence_low_when_similarity_weak():
    assert _confidence_label(0.2, False) == "Low"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_answer_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.answer_engine'`

- [ ] **Step 3: Write `rag/answer_engine.py`**

```python
from dataclasses import dataclass

import google.generativeai as genai

import config
from rag.retriever import RetrievalResult

genai.configure(api_key=config.GEMINI_API_KEY)

NO_INFO_MESSAGE = "No matching information found in the corpus."
FALLBACK_MESSAGE = "I couldn't generate an answer right now. Please try again."


@dataclass
class Citation:
    doc_id: str
    page: int


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    confidence: str


def _confidence_label(top_similarity: float, graph_corroborates: bool) -> str:
    if top_similarity >= 0.75 and graph_corroborates:
        return "High"
    if top_similarity >= 0.5:
        return "Medium"
    return "Low"


def generate_answer(question: str, retrieval: RetrievalResult) -> AnswerResult:
    if not retrieval.chunks or retrieval.chunks[0].similarity < config.SIMILARITY_THRESHOLD:
        return AnswerResult(answer=NO_INFO_MESSAGE, citations=[], confidence="Low")

    context_text = "\n\n".join(
        f"[Source: {c.doc_id}, p.{c.page}]\n{c.text}" for c in retrieval.chunks
    )
    prompt = (
        "You are an industrial knowledge copilot. Answer the question using ONLY the "
        "context below. Cite sources inline using the exact format [Source: doc_id, p.page] "
        "for every claim you make. If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )

    try:
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        response = model.generate_content(prompt)
        answer_text = response.text
    except Exception:
        return AnswerResult(answer=FALLBACK_MESSAGE, citations=[], confidence="Low")

    citations = [Citation(doc_id=c.doc_id, page=c.page) for c in retrieval.chunks]
    confidence = _confidence_label(retrieval.chunks[0].similarity, bool(retrieval.graph_entities))
    return AnswerResult(answer=answer_text, citations=citations, confidence=confidence)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_answer_engine.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add rag/answer_engine.py tests/test_answer_engine.py
git commit -m "Add Gemini answer generation with citations and confidence labeling"
```

---

### Task 10: Streamlit Copilot UI

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `retrieve` (Task 8), `generate_answer` (Task 9)
- Produces: a runnable Streamlit app; no other module depends on this file.

- [ ] **Step 1: Write `app.py`**

```python
import streamlit as st

from rag.answer_engine import generate_answer
from rag.retriever import retrieve

st.set_page_config(page_title="Industrial Knowledge Copilot", layout="centered")
st.title("Industrial Knowledge Copilot")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask about equipment, procedures, maintenance history...")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["citations"]:
            st.caption("Sources: " + ", ".join(f"{c.doc_id} (p.{c.page})" for c in entry["citations"]))
        st.caption(f"Confidence: {entry['confidence']}")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.spinner("Searching the knowledge corpus..."):
        retrieval = retrieve(question)
        result = generate_answer(question, retrieval)
    with st.chat_message("assistant"):
        st.write(result.answer)
        if result.citations:
            st.caption("Sources: " + ", ".join(f"{c.doc_id} (p.{c.page})" for c in result.citations))
        st.caption(f"Confidence: {result.confidence}")
        if retrieval.graph_entities:
            with st.expander("Related entities (knowledge graph)"):
                st.write(", ".join(retrieval.graph_entities))
    st.session_state.history.append({
        "question": question,
        "answer": result.answer,
        "citations": result.citations,
        "confidence": result.confidence,
    })
```

- [ ] **Step 2: Run the app and verify manually**

Run: `streamlit run app.py`
Expected: browser opens to a chat UI; asking "Who performed maintenance on P-101?" returns an answer mentioning Rajesh Kumar, with a "Sources:" caption and a "Confidence:" caption

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "Add Streamlit copilot chat UI"
```

---

### Task 11: Benchmark Q&A Set and Smoke Test

**Files:**
- Create: `benchmark/qa_set.json`
- Create: `benchmark/run_benchmark.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: `ingest_all` (Task 7), `retrieve` (Task 8), `generate_answer` (Task 9)
- Produces: a benchmark report script and one pytest smoke test; nothing else depends on these.

- [ ] **Step 1: Write `benchmark/qa_set.json`**

```json
[
  {"question": "Who performed the seal replacement on pump P-101?", "expected_substring": "Rajesh Kumar"},
  {"question": "What work order covers the tube bundle cleaning on HX-204?", "expected_substring": "WO-2045"},
  {"question": "Which regulatory standard governs preventive maintenance of P-101?", "expected_substring": "OISD-STD-118"},
  {"question": "Who inspected the heat exchanger HX-204?", "expected_substring": "Anita Sharma"},
  {"question": "What regulatory reference applies to inspection of HX-204?", "expected_substring": "PESO-CCE-2019"},
  {"question": "Who inspected the pressure relief valve on V-301?", "expected_substring": "Mohammed Iqbal"},
  {"question": "When was pump P-101 installed?", "expected_substring": "2018-06-01"},
  {"question": "What is equipment C-102?", "expected_substring": "Compressor"},
  {"question": "What statutory act covers dangerous occurrences on pressure vessels like V-301?", "expected_substring": "Factory-Act-1948-Sec-87"},
  {"question": "What date was the pump seal replaced under WO-2044?", "expected_substring": "2026-03-14"}
]
```

- [ ] **Step 2: Write `benchmark/run_benchmark.py`**

```python
import json
from pathlib import Path

from rag.answer_engine import generate_answer
from rag.retriever import retrieve

QA_SET_PATH = Path(__file__).parent / "qa_set.json"


def run() -> None:
    qa_pairs = json.loads(QA_SET_PATH.read_text())
    passed = 0
    for pair in qa_pairs:
        retrieval = retrieve(pair["question"])
        result = generate_answer(pair["question"], retrieval)
        hit = pair["expected_substring"].lower() in result.answer.lower()
        passed += int(hit)
        status = "PASS" if hit else "FAIL"
        print(f"[{status}] Q: {pair['question']}")
        print(f"   Expected substring: {pair['expected_substring']}")
        print(f"   Answer: {result.answer}")
        print(f"   Confidence: {result.confidence}\n")
    print(f"Benchmark result: {passed}/{len(qa_pairs)} passed")


if __name__ == "__main__":
    run()
```

- [ ] **Step 3: Write `tests/test_smoke.py`**

```python
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
```

- [ ] **Step 4: Run the smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS (requires `GEMINI_API_KEY` set and Task 2's synthetic data already generated in `data/incoming/`)

- [ ] **Step 5: Run the full benchmark**

Run: `python -m benchmark.run_benchmark`
Expected: prints a PASS/FAIL line per question plus a final `Benchmark result: N/10 passed` summary

- [ ] **Step 6: Commit**

```bash
git add benchmark/qa_set.json benchmark/run_benchmark.py tests/test_smoke.py
git commit -m "Add benchmark Q&A set, benchmark runner, and end-to-end smoke test"
```

---

### Task 12: README and Final End-to-End Verification

**Files:**
- Create: `README.md` (overwrite existing 1-line stub)

**Interfaces:**
- Consumes: nothing (documentation only)
- Produces: nothing consumed by other tasks — this is the final task.

- [ ] **Step 1: Write `README.md`**

```markdown
# Industrial Knowledge Copilot

A RAG-powered conversational copilot over industrial documents (SOPs, work orders,
inspection reports, equipment master data, regulatory references), backed by a
knowledge graph that links equipment, personnel, dates, and regulatory references
across documents.

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`
3. Generate the synthetic document corpus: `python -m scripts.generate_synthetic_data`
4. Run ingestion: `python -m ingestion.pipeline`
5. Launch the copilot: `streamlit run app.py`

## Testing

- Unit tests: `pytest tests/ -v` (skips the live smoke test if `GEMINI_API_KEY` is unset)
- Benchmark: `python -m benchmark.run_benchmark`

## Architecture

See `docs/superpowers/specs/2026-07-12-industrial-knowledge-copilot-design.md` for the
full design spec.
```

- [ ] **Step 2: Full manual end-to-end verification**

Run in order:
```bash
python -m scripts.generate_synthetic_data
python -m ingestion.pipeline
pytest tests/ -v
python -m benchmark.run_benchmark
```
Expected: synthetic corpus generated, ingestion reports all 5 files ingested with no skipped entries, all pytest tests pass, benchmark shows a majority of the 10 questions passing (some wording variance from Gemini is acceptable — this is a benchmark signal, not a hard gate)

Then run `streamlit run app.py` and manually ask 2-3 questions from `benchmark/qa_set.json` in the browser, confirming citations and confidence labels render correctly.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Add project README with setup and testing instructions"
```
