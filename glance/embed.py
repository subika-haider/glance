"""
Embedding interface and implementations.
- `Embedder`: Protocol (interface) all embedders must satisfy.
- `CLIPEmbedder`: CLIP (clip-ViT-B-32) embedder — maps both text and images into the same
  512-dim vector space, enabling cross-modal search from a single query.
- `_is_cached`: helper to detect whether a HuggingFace model is already downloaded.
"""

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image
from sentence_transformers import SentenceTransformer


@runtime_checkable  # allows isinstance(obj, Embedder) checks at runtime
class Embedder(Protocol):
    def embed_text(self, text: str) -> list[float]: ...
    def embed_image(self, path: Path) -> list[float]: ...

    @property
    def dim(self) -> int: ...


def _is_cached(model_name: str) -> bool:
    # HuggingFace stores models under ~/.cache/huggingface/hub/ by default,
    # overridable via HF_HOME or HUGGINGFACE_HUB_CACHE env vars.
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    slug = "models--" + model_name.replace("/", "--")
    return (hf_cache / slug).exists()


class CLIPEmbedder:
    # clip-ViT-B-32 puts text and images in the same 512-dim vector space,
    # which is the core trick that lets a text query find images and vice versa.
    MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
    _DIM = 512

    def __init__(self) -> None:
        if not _is_cached(self.MODEL_NAME):
            print("first run: downloading CLIP model (~338 MB), this may take a few minutes...")
        else:
            print("loading CLIP model...")
        self._model = SentenceTransformer(self.MODEL_NAME)

    def embed_text(self, text: str) -> list[float]:
        # tolist() converts numpy array to plain Python floats for chromadb
        return self._model.encode(text, convert_to_tensor=False).tolist()

    def embed_image(self, path: Path) -> list[float]:
        img = Image.open(path).convert("RGB")
        return self._model.encode(img, convert_to_tensor=False).tolist()

    @property
    def dim(self) -> int:
        return self._DIM
