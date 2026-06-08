from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image
from sentence_transformers import SentenceTransformer


@runtime_checkable # python 3.8+ decorator to check if class implements protocol at runtime; dont discard at runtime
class Embedder(Protocol):
    def embed_text(self, text: str) -> list[float]: ...
    def embed_image(self, path: Path) -> list[float]: ...

    @property # turns method into readonly attribute
    def dim(self) -> int: ...


class CLIPEmbedder:
    # clip-ViT-B-32 puts text and images in the same 512-dim vector space,
    # which is the core trick that lets one query find both modalities.
    MODEL_NAME = "clip-ViT-B-32"
    _DIM = 512

    def __init__(self) -> None:
        # sentence-transformers checks ~/.cache/huggingface/hub/ first;
        # downloads ~338 MB from HuggingFace on first run only.
        print("loading CLIP model (first run downloads ~338 MB)...")
        self._model = SentenceTransformer(self.MODEL_NAME)

    def embed_text(self, text: str) -> list[float]:
        # encode_text returns a numpy array; tolist() gives plain Python floats
        # that chromadb can store without extra deps.
        return self._model.encode(text, convert_to_tensor=False).tolist()

    def embed_image(self, path: Path) -> list[float]:
        img = Image.open(path).convert("RGB")
        return self._model.encode(img, convert_to_tensor=False).tolist()

    @property
    def dim(self) -> int:
        return self._DIM
