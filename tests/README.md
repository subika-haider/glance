# Running Tests

This directory contains the unit tests for the `glance` application.

## Running Tests with `uv`

Because `pytest` is declared in the `dev` dependency group within `pyproject.toml`, you need to specify the `--group dev` flag when running tests:

```bash
uv run --group dev pytest
```

### Useful Pytest Options

* **Run a specific test file:**
  ```bash
  uv run --group dev pytest tests/test_store.py
  ```

* **Run a specific test function:**
  ```bash
  uv run --group dev pytest tests/test_store.py -k "test_query_exact_match_scores_near_one"
  ```

* **Show output details (verbose):**
  ```bash
  uv run --group dev pytest -v
  ```
