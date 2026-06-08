"""Configuration settings and environment checks for the Glance CLI."""

import sys
from pathlib import Path

# XDG user data dir — separate from the installed package code so the index
# survives uv reinstalls/upgrades. Created on first `glance add` run.
STORAGE_DIR = Path.home() / ".local" / "share" / "glance"
CHROMA_DIR = STORAGE_DIR / "chroma"

# Phase 1 uses a single collection for both images and text (both embedded with CLIP). 
# Phase 2 uses a separate collection for text (embedded with BGE) n seperates model responsbilities.
COLLECTION_ITEMS = "items"

# CLIP cosine similarity sits low (~0.2–0.4) due to the modality gap, so 0.22
# is the practical noise floor. BGE scores are higher (0.7–0.9), hence 0.35.
# may need to change based on results.
MIN_IMAGE_SCORE = 0.22
MIN_TEXT_SCORE = 0.35
DEFAULT_N = 10 # confidence scores more important than absolute value; 10 = good baseline.
QUERY_K = 30  # fetch this many candidates internally, return top DEFAULT_N 

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_TEXT_BYTES = 1 * 1024 * 1024    # 1 MB

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv"}

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".env", "dist", ".uv"}

# CLIP's text encoder hard-truncates at 77 tokens; 60/10 gives safe windows with overlap.
CLIP_CHUNK_TOKENS = 60
CLIP_CHUNK_OVERLAP = 10

def warn_if_windows() -> None:
    if sys.platform == "win32":
        print("warning: glance is designed for macOS/Linux. on Windows, use WSL or a Docker container.")
