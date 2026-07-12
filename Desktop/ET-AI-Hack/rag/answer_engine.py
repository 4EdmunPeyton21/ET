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
    if retrieval.graph_context_chunks:
        graph_text = "\n\n".join(
            f"[Related via knowledge graph — Source: {c.doc_id}, p.{c.page}]\n{c.text}"
            for c in retrieval.graph_context_chunks
        )
        context_text = f"{context_text}\n\n{graph_text}"
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

    citations = []
    seen_sources = set()
    for c in retrieval.chunks + retrieval.graph_context_chunks:
        key = (c.doc_id, c.page)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        citations.append(Citation(doc_id=c.doc_id, page=c.page))
    confidence = _confidence_label(retrieval.chunks[0].similarity, bool(retrieval.graph_entities))
    return AnswerResult(answer=answer_text, citations=citations, confidence=confidence)
