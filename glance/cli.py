"""
CLI entry point for glance.
- Defines all user-facing commands: add, search, ls, rm, status, reindex.
- Uses Typer for command parsing and Rich for output rendering.
- Currently wired to a mock backend; TODO markers show where real calls go.
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
    MIN_TEXT_SCORE,
    STORAGE_DIR,
    warn_if_windows,
)

app = typer.Typer(help="Semantic search for your personal files.")
console = Console()


@app.callback()
def _startup() -> None:
    warn_if_windows()


# --- mock backend (replaced once store/ingest/search are wired up) ---

def _mock_files(path: Path) -> list[dict]:
    return [
        {"path": str(path / "cat.jpg"), "type": "image"},
        {"path": str(path / "notes.md"), "type": "text"},
        {"path": str(path / "kitten.png"), "type": "image"},
        {"path": str(path / "README.md"), "type": "text"},
    ]

def _mock_results() -> list[dict]:
    return [
        {"path": "~/photos/cat.jpg", "type": "image", "score": 0.31, "snippet": None},
        {"path": "~/notes/felines.md", "type": "text", "score": 0.28, "snippet": "Domestic cats are obligate carnivores that evolved from wild ancestors..."},
        {"path": "~/photos/kitten.png", "type": "image", "score": 0.24, "snippet": None},
    ]

def _mock_indexed() -> list[dict]:
    return [
        {"path": "~/photos/cat.jpg", "type": "image"},
        {"path": "~/notes/felines.md", "type": "text"},
        {"path": "~/photos/kitten.png", "type": "image"},
    ]


@app.command()
def add(
    path: Path = typer.Argument(..., help="File, folder, or glob to index."),
    text_ext: list[str] = typer.Option([], "--text-ext", help="Extra text extensions."),
    no_skip_defaults: bool = typer.Option(False, "--no-skip-defaults", help="Don't skip hidden/cache dirs."),
) -> None:
    """Index a file, folder, or glob."""
    files = _mock_files(path)
    n_images = n_text = 0

    # model is loaded once here before the progress bar so the download message
    # (if first run) appears cleanly above the indexing progress.
    # TODO: replace with real CLIPEmbedder() instantiation
    console.print("[dim]loading model...[/dim]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[fname]}"),
        console=console,
    ) as progress:
        task = progress.add_task("indexing", total=len(files), fname="")
        for f in files:
            progress.update(task, fname=Path(f["path"]).name)
            # TODO: ingest.process(f) + embedder.embed + store.add
            if f["type"] == "image":
                n_images += 1
            else:
                n_text += 1
            progress.advance(task)

    console.print(f"indexed {n_images + n_text} items ({n_images} images, {n_text} text)")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query."),
    n: int = typer.Option(DEFAULT_N, "-n", "--num", help="Number of results."),
    type_filter: Optional[str] = typer.Option(None, "--type", help="Restrict to: image or text."),
    min_image_score: float = typer.Option(MIN_IMAGE_SCORE, "--min-image-score"),
    min_text_score: float = typer.Option(MIN_TEXT_SCORE, "--min-text-score"),
    no_score: bool = typer.Option(False, "--no-score", help="Hide the score column."),
) -> None:
    """Search across all indexed content."""
    results = _mock_results()
    if type_filter:
        results = [r for r in results if r["type"] == type_filter]
    results = results[:n]

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
        row = [str(i), r["path"], r["type"]]
        if not no_score:
            row.append(f"{r['score']:.2f}")
        table.add_row(*row)

        # snippet row: indented under path, other columns blank
        if r["snippet"]:
            snippet = r["snippet"][:80]
            snippet_row = ["", f"  [dim]> {snippet}[/dim]", ""]
            if not no_score:
                snippet_row.append("")
            table.add_row(*snippet_row)

    console.print(table)


@app.command()
def ls(
    type_filter: Optional[str] = typer.Option(None, "--type", help="Filter by: image or text."),
) -> None:
    """List all indexed items."""
    items = _mock_indexed()
    if type_filter:
        items = [item for item in items if item["type"] == type_filter]

    if not items:
        console.print("no indexed items")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Path")
    table.add_column("Type", width=6)
    for item in items:
        table.add_row(item["path"], item["type"])
    console.print(table)


@app.command()
def rm(
    path: str = typer.Argument(..., help="Path to remove from the index."),
) -> None:
    """Remove items matching path from the index."""
    typer.confirm(f"remove '{path}' from the index?", abort=True)
    # TODO: store.delete(path)
    console.print("removed 1 item")


@app.command()
def status() -> None:
    """Show index stats: item counts, storage location, model."""
    # TODO: pull real counts from store
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("key", style="bold", width=10)
    table.add_column("value")
    table.add_row("storage", str(STORAGE_DIR))
    table.add_row("images", "0")
    table.add_row("text", "0")
    table.add_row("model", "clip-ViT-B-32")
    console.print(table)


@app.command()
def reindex(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Drop the index and re-embed everything from tracked paths."""
    if not yes:
        typer.confirm("reindex will drop and re-embed all items. continue?", abort=True)
    # TODO: store.clear() + re-run ingest on all tracked paths
    console.print("reindex complete")
