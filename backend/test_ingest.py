"""
test_ingest.py -- Smoke test for POST /ingest endpoint.

Usage (from backend/ directory, with the server running):
    python test_ingest.py
"""

import sys
import requests
from pathlib import Path

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
INGEST_URL = f"{BASE_URL}/ingest"
DATA_DIR = Path(__file__).parent / "data" / "sample_documents"

# All 7 sample documents
SAMPLE_FILES = sorted(DATA_DIR.glob("*.txt"))


def main():
    print("=" * 60)
    print(f"  Smoke test: POST {INGEST_URL}")
    print("=" * 60)

    # --- Health check ---
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"[health] {health.status_code} -> {health.json()}")
    except Exception as e:
        print(f"[ERROR] Could not reach server at {BASE_URL}: {e}")
        print("Make sure the server is running: uvicorn main:app --reload")
        sys.exit(1)

    print("-" * 60)
    print(f"Found {len(SAMPLE_FILES)} sample files in {DATA_DIR}:")
    for fp in SAMPLE_FILES:
        print(f"  - {fp.name} ({fp.stat().st_size:,} bytes)")
    print("-" * 60)

    if not SAMPLE_FILES:
        print("[ERROR] No .txt files found in data/sample_documents/.")
        sys.exit(1)

    # --- Build multipart payload ---
    file_handles = []
    files_param = []
    for fp in SAMPLE_FILES:
        fh = open(fp, "rb")
        file_handles.append(fh)
        files_param.append(("files", (fp.name, fh, "text/plain")))

    print(f"Uploading {len(files_param)} files to {INGEST_URL} ...")
    print("-" * 60)

    try:
        response = requests.post(INGEST_URL, files=files_param, timeout=120)
    finally:
        for fh in file_handles:
            fh.close()

    print(f"HTTP {response.status_code}")
    print("-" * 60)

    if response.status_code != 200:
        print("[ERROR] Non-200 response:")
        print(response.text[:500])
        sys.exit(1)

    data = response.json()
    print(f"documents_processed: {data['documents_processed']}")
    print("-" * 60)

    all_ok = True
    for r in data["results"]:
        tag = "[OK]  " if r["status"] == "success" else "[FAIL]"
        print(f"{tag} {r['filename']}")
        print(f"       status      : {r['status']}")
        print(f"       text_length : {r['text_length']:,} chars")
        if r["status"] == "success":
            preview = r["preview"].replace("\n", " ").strip()
            print(f"       preview     : {preview[:100]}...")
        else:
            print(f"       error_msg   : {r['error_msg']}")
            all_ok = False
        print()

    print("=" * 60)
    if all_ok:
        print(f"  [PASS] All {data['documents_processed']} files processed -- smoke test PASSED")
    else:
        print("  [FAIL] Some files had errors -- check output above")
    print("=" * 60)

    total_chars = sum(r["text_length"] for r in data["results"])
    print(f"\n  Total text extracted: {total_chars:,} characters across {data['documents_processed']} documents")


if __name__ == "__main__":
    main()
