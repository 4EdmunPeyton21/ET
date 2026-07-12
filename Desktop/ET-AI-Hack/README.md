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
