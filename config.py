from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Models
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

# Storage
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
FAISS_INDEX_DIR = BASE_DIR / "data" / "faiss"
CHECKPOINT_DB = BASE_DIR / "data" / "checkpoints.db"

# Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))


def validate_config() -> None:
    missing = []

    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    if not SERPAPI_API_KEY:
        missing.append("SERPAPI_API_KEY")

    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}"
        )


# Create local storage directories if they don't exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)