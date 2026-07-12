import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INCOMING_DIR = DATA_DIR / "incoming"
CHROMA_DIR = DATA_DIR / "chroma"
GRAPH_PATH = DATA_DIR / "graph" / "knowledge_graph.pkl"

CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
