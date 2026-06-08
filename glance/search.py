"""
Search orchestration for Phase 1 (CLIP-only, single collection).
- `Result`: dataclass returned to cli.py — best-scoring hit per file, with extra_matches count.
- `run`: main entry point. Encodes query → queries store → filters by threshold →
  deduplicates by path (best chunk wins, extras counted) → returns top N results.
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
    # encode the query with CLIP — same space as both text and image embeddings
    query_vec = embedder.embed_text(query)

    # fetch more candidates than needed internally to give dedup room to work
    raw_hits = store.query(query_vec, k=QUERY_K)

    # apply score threshold — drop noisy results below the confidence floor
    hits = [h for h in raw_hits if h.score >= min_score]

    # apply type filter if requested
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

    # write extra counts back onto the result objects
    for path, result in best.items():
        result.extra_matches = extras[path]

    # sort by score descending and return top N
    ranked = sorted(best.values(), key=lambda r: r.score, reverse=True)
    return ranked[:n]
