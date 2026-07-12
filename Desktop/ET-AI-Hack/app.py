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
