"""
CLI entry point for glance.
- Defines all user-facing commands: add, search, ls, rm, status, clear.
- Uses Typer for command parsing and Rich for output rendering.
- The CLIP model and Chroma store are only created when a command actually needs them
  (not at startup). This means `glance status` or `glance clear` never trigger a model
  load — only commands that embed something (add, search) do.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

from glance.config import (
    DEFAULT_N,
    MIN_IMAGE_SCORE,
    STORAGE_DIR,
    warn_if_windows,
)

app = typer.Typer(help="Semantic search for your personal files.")
console = Console()

# module-level cache so embedder and store are each created at most once per process
_embedder = None
_store = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from glance.embed import CLIPEmbedder
        _embedder = CLIPEmbedder()
    return _embedder


def _get_store():
    global _store
    if _store is None:
        from glance.store import ChromaStore
        _store = ChromaStore()
    return _store


@app.callback()
def _startup() -> None:
    warn_if_windows()


@app.command()
def add(
    path: Path = typer.Argument(..., help="File, folder, or glob to index."),
    text_ext: list[str] = typer.Option([], "--text-ext", help="Extra text extensions."),
    no_skip_defaults: bool = typer.Option(False, "--no-skip-defaults", help="Don't skip hidden/cache dirs."),
) -> None:
    """Index a file, folder, or glob."""
    from glance.ingest import discover, make_items

    # load model before progress bar so download message (if any) appears cleanly above it
    embedder = _get_embedder()
    store = _get_store()

    files = list(discover(path, extra_text_exts=set(text_ext), skip_defaults=not no_skip_defaults))

    if not files:
        console.print("no indexable files found")
        return

    if len(files) > 500:
        console.print(f"warning: indexing {len(files)} files — this may take a while")

    n_images = n_text = n_skipped = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[fname]}"),
        console=console,
    ) as progress:
        task = progress.add_task("indexing", total=len(files), fname="")

        for file in files:
            progress.update(task, fname=file.name)
            items = make_items(file, embedder)

            for item in items:
                try:
                    if item["type"] == "image":
                        vec = embedder.embed_image(Path(item["path"]))
                        store.add(item["id"], vec, {"path": item["path"], "type": "image"})
                        n_images += 1
                    else:
                        vec = embedder.embed_text(item["text"])
                        store.add(item["id"], vec, {"path": item["path"], "type": "text", "text": item["text"]})
                        n_text += 1
                except Exception as e:
                    console.print(f"skipped {file.name} — {e}")
                    n_skipped += 1

            progress.advance(task)

    summary = f"indexed {n_images + n_text} items ({n_images} images, {n_text} text)"
    if n_skipped:
        summary += f", {n_skipped} skipped"
    console.print(summary)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query."),
    n: int = typer.Option(DEFAULT_N, "-n", "--num", help="Number of results."),
    type_filter: Optional[str] = typer.Option(None, "--type", help="Restrict to: image or text."),
    min_image_score: float = typer.Option(MIN_IMAGE_SCORE, "--min-image-score"),
    no_score: bool = typer.Option(False, "--no-score", help="Hide the score column."),
) -> None:
    """Search across all indexed content."""
    import glance.search as search_mod

    results = search_mod.run(
        query=query,
        store=_get_store(),
        embedder=_get_embedder(),
        n=n,
        min_score=min_image_score,
        type_filter=type_filter,
    )

    if not results:
        console.print("no results")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Path")
    table.add_column("Type", width=6)
    if not no_score:
        table.add_column("Score", width=6)

    for i, r in enumerate(results, 1):
        row = [str(i), r.path, r.type]
        if not no_score:
            row.append(f"{r.score:.2f}")
        table.add_row(*row)

        if r.snippet:
            snippet = r.snippet[:80]
            extra = f"  [dim]+{r.extra_matches} more[/dim]" if r.extra_matches else ""
            snippet_row = ["", f"  [dim]> {snippet}[/dim]{extra}", ""]
            if not no_score:
                snippet_row.append("")
            table.add_row(*snippet_row)

    console.print(table)


@app.command()
def ls(
    type_filter: Optional[str] = typer.Option(None, "--type", help="Filter by: image or text."),
) -> None:
    """List all indexed items."""
    items = _get_store().list(type_filter=type_filter)

    if not items:
        console.print("no indexed items")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Path")
    table.add_column("Type", width=6)
    for item in items:
        table.add_row(item.path, item.type)
    console.print(table)


@app.command()
def rm(
    path: str = typer.Argument(..., help="Path to remove from the index."),
) -> None:
    """Remove items matching path from the index."""
    typer.confirm(f"remove '{path}' from the index?", abort=True)
    removed = _get_store().delete_by_path(path)
    console.print(f"removed {removed} item{'s' if removed != 1 else ''}")


@app.command()
def status() -> None:
    """Show index stats: item counts, storage location, model."""
    counts = _get_store().count()
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("key", style="bold", width=10)
    table.add_column("value")
    table.add_row("storage", str(STORAGE_DIR))
    table.add_row("images", str(counts["image"]))
    table.add_row("text", str(counts["text"]))
    table.add_row("model", "clip-ViT-B-32")
    console.print(table)


@app.command()
def clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Drop the entire index. Re-run `glance add` to rebuild."""
    if not yes:
        typer.confirm("this will delete all indexed data. continue?", abort=True)
    _get_store().clear()
    console.print("index cleared. run `glance add <path>` to rebuild.")
