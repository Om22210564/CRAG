from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval import build_vector_store


if __name__ == "__main__":
    build_vector_store()
    print("FAISS index created successfully.")