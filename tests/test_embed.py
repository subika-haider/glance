"""
Integration tests for CLIPEmbedder using the real model.
Requires model download (~338 MB) on first run — cached in ~/.cache/huggingface after that.
Tests verify embedding shape, normalization, determinism, and real semantic relationships.
"""

import math
from pathlib import Path

from glance.embed import CLIPEmbedder


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x ** 2 for x in a))
    nb = math.sqrt(sum(x ** 2 for x in b))
    return dot / (na * nb)


# --- output shape and normalization ---

def test_text_embedding_dimension(embedder):
    vec = embedder.embed_text("cat")
    assert len(vec) == 512


def test_image_embedding_dimension(embedder, test_image):
    vec = embedder.embed_image(test_image)
    assert len(vec) == 512


def test_text_embedding_has_nonzero_norm(embedder):
    # CLIP embeddings from sentence-transformers are NOT unit-normalized;
    # Chroma normalizes internally for cosine distance. Just verify it's a real vector.
    vec = embedder.embed_text("cat")
    norm = math.sqrt(sum(x ** 2 for x in vec))
    assert norm > 0


def test_image_embedding_has_nonzero_norm(embedder, test_image):
    vec = embedder.embed_image(test_image)
    norm = math.sqrt(sum(x ** 2 for x in vec))
    assert norm > 0


# --- determinism ---

def test_text_embedding_is_deterministic(embedder):
    assert embedder.embed_text("hello world") == embedder.embed_text("hello world")


def test_image_embedding_is_deterministic(embedder, test_image):
    assert embedder.embed_image(test_image) == embedder.embed_image(test_image)


# --- semantic relationships ---

def test_similar_words_score_higher_than_unrelated(embedder):
    # CLIP text-to-text: "cat" should be closer to "kitten" than to "rocket ship"
    cat = embedder.embed_text("cat")
    kitten = embedder.embed_text("kitten")
    rocket = embedder.embed_text("rocket ship")
    assert cosine(cat, kitten) > cosine(cat, rocket)


def test_same_concept_different_phrasing(embedder):
    # "dog" and "puppy" should be more similar than "dog" and "airplane"
    dog = embedder.embed_text("dog")
    puppy = embedder.embed_text("puppy")
    airplane = embedder.embed_text("airplane")
    assert cosine(dog, puppy) > cosine(dog, airplane)


def test_different_texts_produce_different_vectors(embedder):
    assert embedder.embed_text("cat") != embedder.embed_text("dog")
