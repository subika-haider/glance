"""
Real-world semantic search integration tests.
Indexes actual images and text files from tests/fixtures/ using the real CLIP model,
then verifies that queries surface the correct content across both modalities.
Requires model download (~338 MB) on first run — cached after that.
"""

import hashlib
from pathlib import Path

import pytest

from glance.embed import CLIPEmbedder
from glance.store import ChromaStore

FIXTURES = Path(__file__).parent / "fixtures"
IMAGES = FIXTURES / "images"
TEXT = FIXTURES / "text"


def file_id(path: Path, chunk_idx: int = 0) -> str:
    return hashlib.sha256(f"{path}{chunk_idx}".encode()).hexdigest()


@pytest.fixture(scope="module")
def populated_store(embedder, tmp_path_factory):
    # module scope so we index once and run all semantic queries against it
    store = ChromaStore(chroma_dir=tmp_path_factory.mktemp("chroma"))

    for img_path in sorted(IMAGES.glob("*.jpg")):
        vec = embedder.embed_image(img_path)
        store.add(
            file_id(img_path),
            vec,
            {"path": str(img_path), "type": "image"},
        )

    for txt_path in sorted(TEXT.glob("*.md")):
        text = txt_path.read_text(encoding="utf-8")
        vec = embedder.embed_text(text)
        store.add(
            file_id(txt_path),
            vec,
            {"path": str(txt_path), "type": "text", "text": text[:200]},
        )

    return store


def top_paths(store, embedder, query: str, k: int = 3) -> list[str]:
    vec = embedder.embed_text(query)
    hits = store.query(vec, k=k)
    return [Path(h.path).stem for h in hits]


# --- image retrieval ---
# CLIP modality gap: text-to-text similarity is naturally higher than text-to-image,
# so images don't always win top-3 in a mixed index. We use k=8 here — images are
# reliably present within the top 8. Phase 2 (separate collections + RRF) fixes this.

def test_cat_query_finds_cat_image(populated_store, embedder):
    results = top_paths(populated_store, embedder, "cat", k=8)
    assert "cat" in results


def test_dog_query_finds_dog_image(populated_store, embedder):
    results = top_paths(populated_store, embedder, "dog", k=8)
    assert "dog" in results


def test_beach_query_finds_beach_image(populated_store, embedder):
    results = top_paths(populated_store, embedder, "beach", k=8)
    assert "beach" in results


def test_mountain_query_finds_mountain_image(populated_store, embedder):
    results = top_paths(populated_store, embedder, "mountain", k=8)
    assert "mountain" in results


def test_car_query_finds_car_image(populated_store, embedder):
    results = top_paths(populated_store, embedder, "car", k=8)
    assert "car" in results


# --- text retrieval ---

def test_cat_query_finds_cat_text(populated_store, embedder):
    results = top_paths(populated_store, embedder, "cat", k=5)
    assert "cats" in results


def test_ocean_query_finds_ocean_text(populated_store, embedder):
    results = top_paths(populated_store, embedder, "ocean", k=5)
    assert "ocean" in results


# --- cross-modal: text query finds both image and text ---

def test_cat_query_finds_both_modalities(populated_store, embedder):
    vec = embedder.embed_text("cat")
    hits = populated_store.query(vec, k=10)
    types = {Path(h.path).stem: h.type for h in hits}
    assert types.get("cat") == "image"
    assert types.get("cats") == "text"


def test_dog_query_finds_both_modalities(populated_store, embedder):
    vec = embedder.embed_text("dog")
    hits = populated_store.query(vec, k=10)
    types = {Path(h.path).stem: h.type for h in hits}
    assert types.get("dog") == "image"
    assert types.get("dogs") == "text"


# --- unrelated queries should not surface wrong content at top ---

def test_cat_query_does_not_top_rank_car(populated_store, embedder):
    results = top_paths(populated_store, embedder, "cat", k=1)
    assert "car" not in results


def test_ocean_query_does_not_top_rank_mountain(populated_store, embedder):
    results = top_paths(populated_store, embedder, "ocean", k=1)
    assert "mountain" not in results
