"""
Search orchestration for Phase 1 (CLIP-only, single collection).
- `Result`: dataclass returned to cli.py — best-scoring hit per file, with extra_matches count.
- `run`: main entry point. Runs two queries — raw query for text, "a photo of a {query}"
  for images — then merges results. This closes the CLIP modality gap where text-to-text
  similarity (0.7-0.9) would otherwise swamp text-to-image (0.2-0.4) in a mixed index.
Phase 2 will extend this file to dual-index (CLIP + BGE) with RRF merging.
"""

from dataclasses import dataclass

from glance.config import MIN_IMAGE_SCORE, QUERY_K
# MIN_TEXT_SCORE imported here for Phase 2 — Phase 1 uses a single CLIP threshold for both modalities
from glance.embed import Embedder
from glance.store import ChromaStore


@dataclass
class Result:
    path: str
    type: str           # "image" or "text"
    score: float        # cosine similarity of the best-matching chunk
    snippet: str | None # best chunk's text, None for images
    extra_matches: int  # how many other chunks from this file also passed the threshold


def run(
    query: str,
    store: ChromaStore,
    embedder: Embedder,
    n: int,
    min_score: float = MIN_IMAGE_SCORE,  # Phase 1: one CLIP threshold for all types. Phase 2: split into min_image_score / min_text_score per index.
    type_filter: str | None = None,
) -> list[Result]:
    # text query: raw string works well for text-to-text similarity
    text_vec = embedder.embed_text(query)

    # image query: "a photo of a {query}" is a standard CLIP trick — the model was trained
    # on image captions, so this prefix dramatically improves text-to-image retrieval
    # and closes the modality gap vs text results in a mixed index.
    image_vec = embedder.embed_text(f"a photo of a {query}")

    text_hits = store.query(text_vec, k=QUERY_K)
    image_hits = store.query(image_vec, k=QUERY_K)

    # separate by type — use the right query vector's scores for each modality
    hits = (
        [h for h in text_hits if h.type == "text"] +
        [h for h in image_hits if h.type == "image"]
    )

    # apply score threshold and optional type filter
    hits = [h for h in hits if h.score >= min_score]
    if type_filter:
        hits = [h for h in hits if h.type == type_filter]

    # deduplicate by path: keep best-scoring chunk per file, count the rest.
    # this prevents a long text file from dominating the result list with many chunks.
    best: dict[str, Result] = {}
    extras: dict[str, int] = {}

    for hit in hits:
        if hit.path not in best:
            best[hit.path] = Result(
                path=hit.path,
                type=hit.type,
                score=hit.score,
                snippet=hit.snippet,
                extra_matches=0,
            )
            extras[hit.path] = 0
        else:
            # another chunk from the same file — count it for the (+N more) annotation
            extras[hit.path] += 1

    for path, result in best.items():
        result.extra_matches = extras[path]

    # when no type filter is set, interleave image and text results by rank rather than
    # sorting by raw score — CLIP scores are on different scales per modality (images
    # 0.2-0.4, text 0.7-0.9) so score-sorting would bury all images below all text.
    if type_filter:
        ranked = sorted(best.values(), key=lambda r: r.score, reverse=True)
    else:
        images = sorted([r for r in best.values() if r.type == "image"], key=lambda r: r.score, reverse=True)
        texts = sorted([r for r in best.values() if r.type == "text"], key=lambda r: r.score, reverse=True)
        # round-robin merge: text, image, text, image... so both modalities surface early
        ranked = []
        for t, i in zip(texts, images):
            ranked.append(t)
            ranked.append(i)
        # append any leftovers if one list is longer
        ranked.extend(texts[len(images):])
        ranked.extend(images[len(texts):])

    return ranked[:n]
