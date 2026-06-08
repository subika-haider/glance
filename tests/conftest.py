"""
Shared pytest fixtures.
- `embedder`: session-scoped CLIPEmbedder — loaded once, reused across all tests.
  First run triggers ~338 MB HuggingFace download; subsequent runs use cache.
- `test_image`: a small generated PNG written to tmp_path for image embedding tests.
"""

import pytest
from pathlib import Path
from PIL import Image

from glance.embed import CLIPEmbedder


@pytest.fixture(scope="session")
def embedder():
    # session scope: model loads once for the entire test run, not per test
    return CLIPEmbedder()


@pytest.fixture
def test_image(tmp_path) -> Path:
    # generate a simple 224x224 RGB image — no download needed, just Pillow
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (224, 224), color=(120, 80, 200))
    img.save(img_path)
    return img_path
