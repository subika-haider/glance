from pathlib import Path

STORAGE_DIR = Path.home() / ".local" / "share" / "glance"
CHROMA_DIR = STORAGE_DIR / "chroma"

COLLECTION_ITEMS = "items"

MIN_IMAGE_SCORE = 0.22
MIN_TEXT_SCORE = 0.35
DEFAULT_N = 10
QUERY_K = 30

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TEXT_BYTES = 1 * 1024 * 1024

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv"}

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".env", "dist", ".uv"}

CLIP_CHUNK_TOKENS = 60
CLIP_CHUNK_OVERLAP = 10
