"""
Chroma vector store wrapper.
- `Hit`: result returned from a similarity query (path, type, score, optional snippet).
- `Item`: result returned from a list operation (path, type).
- `Store`: Protocol defining the storage interface.
- `ChromaStore`: persistent Chroma-backed implementation of Store.
- `ChromaStore.clear`: drops and recreates the collection, wiping all indexed data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb

from glance.config import CHROMA_DIR, COLLECTION_ITEMS


@dataclass
class Hit: # represents search result 
    path: str
    type: str       # "image" or "text"
    score: float    # cosine similarity (higher = better match)
    snippet: str | None = None  # best-matching chunk text, text results only, optional


@dataclass
class Item: # represents unique file entry 
    path: str
    type: str


class Store(Protocol): # defines the interface for the storage system 
    def add(self, id: str, embedding: list[float], metadata: dict) -> None: ...
    def query(self, embedding: list[float], k: int) -> list[Hit]: ...
    def delete_by_path(self, path: str) -> int: ...
    def list(self, type_filter: str | None = None) -> list[Item]: ...
    def count(self) -> dict[str, int]: ...


class ChromaStore: # concrete implementation of Store protocol using Chroma
    def __init__(self, chroma_dir: Path | None = None) -> None:
        # chroma_dir can be overridden for testing so tests don't touch ~/.local/share/glance
        dir = chroma_dir or CHROMA_DIR
        dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(dir)) # run in embedded mode w disk persistence. runs in python runtime and is saved to local directory.
        
        # cosine space: distance = 1 - cosine_similarity, so score = 1 - distance.
        # must be set at collection creation; ignored on subsequent get_or_create calls.
        self._col = self._client.get_or_create_collection(
            COLLECTION_ITEMS,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, id: str, embedding: list[float], metadata: dict) -> None: # id is sha hash
        # upsert overwrites if the id already exists — handles re-indexing same chunk
        self._col.upsert(ids=[id], embeddings=[embedding], metadatas=[metadata])

    def query(self, embedding: list[float], k: int) -> list[Hit]:
        results = self._col.query(query_embeddings=[embedding], n_results=k)
        hits = []
        for i, id_ in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = 1.0 - distance  # convert chroma cosine distance to similarity
            # cosine distance ranges [0, 2]. score ranges [-1, 1].
            hits.append(Hit(
                path=meta["path"],
                type=meta["type"],
                score=round(score, 4),
                snippet=meta.get("text"),  # None for images, chunk text for text files
            ))
        return hits

    def delete_by_path(self, path: str) -> int:
        # fetch all ids matching this path (could be multiple chunks), then delete
        existing = self._col.get(where={"path": path})
        ids = existing["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    def list(self, type_filter: str | None = None) -> list[Item]:
        where = {"type": type_filter} if type_filter else None
        results = self._col.get(where=where) if where else self._col.get()
        # deduplicate paths — text files produce multiple chunk rows
        seen: set[str] = set()
        items = []
        for meta in results["metadatas"]:
            path = meta["path"]
            if path not in seen:
                seen.add(path)
                items.append(Item(path=path, type=meta["type"]))
        return items

    def count(self) -> dict[str, int]:
        # returns {"image": N, "text": M} for use in `glance status`
        images = self._col.get(where={"type": "image"})
        texts = self._col.get(where={"type": "text"})
        return {"image": len(images["ids"]), "text": len(texts["ids"])}

    def clear(self) -> None:
        # drop and recreate the collection — wipes all indexed data.
        # used by `glance clear`; user must re-run `glance add` to rebuild.
        self._client.delete_collection(COLLECTION_ITEMS)
        self._col = self._client.get_or_create_collection(
            COLLECTION_ITEMS,
            metadata={"hnsw:space": "cosine"},
        )
