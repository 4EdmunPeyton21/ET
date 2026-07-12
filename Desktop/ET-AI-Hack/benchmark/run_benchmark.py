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
