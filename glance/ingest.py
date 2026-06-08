"""
File discovery, type detection, and ID generation.
- `discover`: walks a path and yields indexable files, skipping hidden/cache dirs and oversized files.
- `detect_type`: returns "image", "text", or None based on file extension.
- `make_id`: deterministic sha256 ID for a file+chunk pair.
- `make_items`: top-level function — given a file path and embedder, returns all ingest-ready items.
  Text chunking is delegated to embedder.chunk_text() since chunk size is model-dependent.
"""

import hashlib
from pathlib import Path
from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from glance.embed import Embedder

from glance.config import (
    IMAGE_EXTENSIONS,
    MAX_IMAGE_BYTES,
    MAX_TEXT_BYTES,
    SKIP_DIRS,
    TEXT_EXTENSIONS,
)


# --- type detection ---

def detect_type(path: Path, extra_exts: set[str] | None = None) -> str | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in TEXT_EXTENSIONS | (extra_exts or set()):
        return "text"
    return None


# --- file discovery ---

def discover(
    root: Path,
    extra_text_exts: set[str] | None = None,
    skip_defaults: bool = True,
) -> Generator[Path, None, None]:
    if root.is_file():
        yield from _check_file(root, extra_text_exts or set())
        return

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # skip hidden files and anything inside a skip dir
        if skip_defaults and any(p.startswith(".") or p in SKIP_DIRS for p in path.parts):
            continue
        yield from _check_file(path, extra_text_exts or set())


def _check_file(path: Path, extra_text_exts: set[str]) -> Generator[Path, None, None]:
    ext = path.suffix.lower()
    is_image = ext in IMAGE_EXTENSIONS
    is_text = ext in TEXT_EXTENSIONS | extra_text_exts

    if not is_image and not is_text:
        return  # silently skip unsupported types (PDFs, binaries, etc.)

    size = path.stat().st_size
    limit = MAX_IMAGE_BYTES if is_image else MAX_TEXT_BYTES
    if size > limit:
        mb = limit // (1024 * 1024)
        print(f"skipped {path.name} — exceeds {mb} MB limit")
        return

    yield path


# --- ID generation ---

def make_id(path: Path, chunk_idx: int) -> str:
    # deterministic: same path + chunk always produces the same id,
    # so re-indexing the same file upserts rather than duplicates rows.
    return hashlib.sha256(f"{path}{chunk_idx}".encode()).hexdigest()


# --- top-level entry point ---

def make_items(path: Path, embedder: "Embedder") -> list[dict]:
    """Returns all ingest-ready dicts for a single file (1 for images, N chunks for text)."""
    ftype = detect_type(path)

    if ftype == "image":
        return [{"id": make_id(path, 0), "path": str(path), "type": "image"}]

    if ftype == "text":
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"skipped {path.name} — could not decode as UTF-8")
            return []

        # chunking delegated to the embedder — it knows its own token limits
        chunks = embedder.chunk_text(text)
        return [
            {
                "id": make_id(path, i),
                "path": str(path),
                "type": "text",
                "chunk_idx": i,
                "text": chunk,
            }
            for i, chunk in enumerate(chunks)
        ]

    return []
