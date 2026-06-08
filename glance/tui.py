"""
Interactive TUI for glance. Launched by `glance` with no subcommand.
Uses Textual for a full-screen app with four modes: search, list, add, status.
Model loads in a background thread behind a loading screen on startup.
Windows users see a modal warning before the main app appears.
"""

import subprocess
import sys
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Middle, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    LoadingIndicator,
    Static,
)


MODES = ("search", "list", "add", "status")


# ── Loading screen ────────────────────────────────────────────────────────────

class LoadingScreen(Screen):
    DEFAULT_CSS = """
    LoadingScreen {
        align: center middle;
    }
    LoadingScreen #load-msg {
        text-align: center;
        margin-bottom: 1;
    }
    """

    def __init__(self, first_run: bool = False) -> None:
        super().__init__()
        self._first_run = first_run

    def compose(self) -> ComposeResult:
        msg = (
            "downloading CLIP model (~338 MB), this may take a few minutes..."
            if self._first_run
            else "loading CLIP model..."
        )
        yield Label(msg, id="load-msg")
        yield LoadingIndicator()


# ── Windows warning ───────────────────────────────────────────────────────────

class WindowsWarningModal(ModalScreen):
    DEFAULT_CSS = """
    WindowsWarningModal {
        align: center middle;
    }
    WindowsWarningModal #dialog {
        padding: 2 4;
        border: solid $warning;
        width: 60;
        height: auto;
        background: $surface;
    }
    WindowsWarningModal #dismiss-hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("enter", "dismiss_modal", "Continue", show=False),
        Binding("escape", "dismiss_modal", "Continue", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("glance is designed for macOS and Linux.")
            yield Label("On Windows, use WSL or a Docker container.")
            yield Label("Press Enter or Esc to continue.", id="dismiss-hint")

    def action_dismiss_modal(self) -> None:
        self.dismiss()


# ── Confirm modal ────────────────────────────────────────────────────────────

class ConfirmModal(ModalScreen):
    """Confirmation dialog — Enter to confirm, Esc to cancel."""

    DEFAULT_CSS = """
    ConfirmModal {
        align: center middle;
    }
    ConfirmModal #dialog {
        padding: 2 4;
        border: solid $error;
        width: 50;
        height: auto;
        background: $surface;
    }
    ConfirmModal #hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._message)
            yield Label("Enter to confirm, Esc to cancel.", id="hint")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


# ── List items ────────────────────────────────────────────────────────────────

class ResultItem(ListItem):
    """Search result row — one per file, includes snippet and extra-match count."""

    def __init__(self, result, rank: int) -> None:
        super().__init__()
        self._result = result
        self._rank = rank

    def compose(self) -> ComposeResult:
        r = self._result
        score_str = f"{r.score:.2f}"
        yield Label(f"{self._rank}. {r.path}  [{r.type}]  {score_str}")
        if r.snippet:
            snippet = r.snippet[:70].replace("\n", " ")
            yield Label(f"  > {snippet}", classes="snippet")
        if r.extra_matches:
            yield Label(f"  +{r.extra_matches} more chunk{'s' if r.extra_matches != 1 else ''}", classes="dim")

    @property
    def result(self):
        return self._result


class FileItem(ListItem):
    """Indexed file row used in list mode."""

    def __init__(self, path: str, ftype: str) -> None:
        super().__init__()
        self._path = path
        self._ftype = ftype

    def compose(self) -> ComposeResult:
        yield Label(f"{self._path}  [{self._ftype}]")

    @property
    def path(self) -> str:
        return self._path


# ── Main app ──────────────────────────────────────────────────────────────────

class GlanceApp(App):
    """glance — unified semantic search for personal files."""

    TITLE = "glance"
    SUB_TITLE = "search"

    CSS = """
    #query-input {
        margin: 0 1;
    }
    #results-list {
        height: 1fr;
    }
    #status-display {
        height: 1fr;
        padding: 1 2;
    }
    #status-bar {
        height: 1;
        background: $surface-darken-1;
        color: $text-muted;
        padding: 0 1;
    }
    .snippet {
        color: $text-muted;
    }
    .dim {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "cycle_mode", "Mode", priority=True),
        Binding("o", "open_file", "Open"),
        Binding("c", "copy_path", "Copy path"),
        Binding("r", "remove_item", "Remove"),
        Binding("i", "filter_images", "Images only"),
        Binding("t", "filter_text", "Text only"),
        Binding("a", "filter_all", "All types"),
        Binding("escape", "unlock", "Back to input"),
    ]

    mode: reactive[str] = reactive("search")
    type_filter: reactive[str | None] = reactive(None)
    locked: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self._embedder = None
        self._store = None
        self._search_timer = None
        self._results: list = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon=" ")
        yield Input(placeholder="search...", id="query-input")
        yield ListView(id="results-list")
        yield Static("", id="status-display")
        yield Label("", id="status-bar")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#status-display").display = False

        # Skip model loading if embedder was injected (e.g. in tests)
        if self._embedder is not None:
            self._on_model_loaded()
            return

        from glance.embed import _is_cached, CLIPEmbedder
        first_run = not _is_cached(CLIPEmbedder.MODEL_NAME)
        await self.push_screen(LoadingScreen(first_run=first_run))
        self._load_model()

    @work(thread=True, exclusive=True)
    def _load_model(self) -> None:
        from glance.embed import CLIPEmbedder
        from glance.store import ChromaStore
        self._embedder = CLIPEmbedder()
        self._store = ChromaStore()
        self.call_from_thread(self._on_model_loaded)

    def on_worker_failed(self, event) -> None:
        # If model loading fails, exit cleanly with an error message
        self.exit(message=f"failed to load model: {event.worker.error}")

    def _on_model_loaded(self) -> None:
        if len(self.screen_stack) > 1:
            self.pop_screen()  # dismiss LoadingScreen
        self.query_one("#query-input", Input).focus()
        self._set_status("ready — type to search")
        if sys.platform == "win32":
            self.push_screen(WindowsWarningModal())

    # ── Input ─────────────────────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        if self.mode != "search" or self._embedder is None:
            return
        if self._search_timer is not None:
            self._search_timer.stop()
        self._search_timer = self.set_timer(0.2, lambda: self._trigger_search(event.value))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.mode == "search":
            if not self.locked and self._results:
                lv = self.query_one("#results-list", ListView)
                self.locked = True
                if lv.index is None:
                    lv.index = 0  # highlight first item when locking
                lv.focus()
        elif self.mode == "add":
            path = event.value.strip()
            if path:
                event.input.value = ""
                self._run_add(path)

    # ── Search ────────────────────────────────────────────────────────────────

    def _trigger_search(self, query: str) -> None:
        if not query.strip():
            self._clear_results()
            return
        self._run_search(query)

    @work(thread=True, exclusive=True)
    def _run_search(self, query: str) -> None:
        import glance.search as search_mod
        results = search_mod.run(
            query=query,
            store=self._store,
            embedder=self._embedder,
            n=10,
            type_filter=self.type_filter,
        )
        self.call_from_thread(self._update_results, results)

    def _update_results(self, results: list) -> None:
        self._results = results
        lv = self.query_one("#results-list", ListView)
        lv.clear()
        if not results:
            self._set_status("no results")
            return
        for i, r in enumerate(results, 1):
            lv.append(ResultItem(r, i))
        count = len(results)
        self._set_status(f"{count} result{'s' if count != 1 else ''}")

    def _clear_results(self) -> None:
        self._results = []
        self.query_one("#results-list", ListView).clear()
        self._set_status("")

    # ── List view selected (Enter on item) ────────────────────────────────────

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, ResultItem):
            self._open_file(event.item.result.path)
        elif isinstance(event.item, FileItem):
            self._open_file(event.item.path)

    # ── Add mode indexing ─────────────────────────────────────────────────────

    @work(thread=True, exclusive=True)
    def _run_add(self, path_str: str) -> None:
        from glance.ingest import discover, make_items
        path = Path(path_str)

        if not path.exists():
            self.call_from_thread(self._set_status, f"path not found: {path_str}")
            return

        files = list(discover(path))
        self.call_from_thread(self._set_status, f"indexing {len(files)} files...")

        n_done = n_skipped = 0
        for file in files:
            items = make_items(file, self._embedder)
            for item in items:
                try:
                    if item["type"] == "image":
                        vec = self._embedder.embed_image(Path(item["path"]))
                        self._store.add(item["id"], vec, {"path": item["path"], "type": "image"})
                    else:
                        vec = self._embedder.embed_text(item["text"])
                        self._store.add(item["id"], vec, {"path": item["path"], "type": "text", "text": item["text"]})
                    n_done += 1
                except Exception:
                    n_skipped += 1

        msg = f"indexed {n_done} item{'s' if n_done != 1 else ''}"
        if n_skipped:
            msg += f", {n_skipped} skipped"
        self.call_from_thread(self._set_status, msg)

    # ── File open / clipboard ─────────────────────────────────────────────────

    def _open_file(self, path: str) -> None:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            elif sys.platform == "win32":
                subprocess.run(["start", path], shell=True, check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
            self._set_status(f"opened {Path(path).name}")
        except Exception as e:
            self._set_status(f"could not open: {e}")

    def _copy_to_clipboard(self, text: str) -> None:
        try:
            if sys.platform == "darwin":
                subprocess.run(["pbcopy"], input=text.encode(), check=True)
                self._set_status(f"copied: {text}")
            elif sys.platform == "win32":
                subprocess.run(["clip"], input=text.encode(), check=True)
                self._set_status(f"copied: {text}")
            else:
                try:
                    subprocess.run(["xclip", "-selection", "clipboard"],
                                   input=text.encode(), check=True)
                    self._set_status(f"copied: {text}")
                except FileNotFoundError:
                    subprocess.run(["xsel", "--clipboard", "--input"],
                                   input=text.encode(), check=True)
                    self._set_status(f"copied: {text}")
        except Exception:
            self._set_status("clipboard unavailable (install xclip or xsel on Linux)")

    def _get_selected_path(self) -> str | None:
        lv = self.query_one("#results-list", ListView)
        child = lv.highlighted_child
        if isinstance(child, ResultItem):
            return child.result.path
        if isinstance(child, FileItem):
            return child.path
        return None

    # ── Key actions ───────────────────────────────────────────────────────────

    def action_unlock(self) -> None:
        self.locked = False
        self.query_one("#query-input", Input).focus()

    def action_open_file(self) -> None:
        path = self._get_selected_path()
        if path:
            self._open_file(path)
        else:
            self._set_status("no item selected")

    def action_copy_path(self) -> None:
        path = self._get_selected_path()
        if path:
            self._copy_to_clipboard(path)
        else:
            self._set_status("no item selected")

    def action_remove_item(self) -> None:
        path = self._get_selected_path()
        if not path:
            self._set_status("no item selected")
            return

        def handle_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            removed = self._store.delete_by_path(path)
            self._set_status(f"removed {removed} item{'s' if removed != 1 else ''}")
            if self.mode == "list":
                self._load_list()
            else:
                query = self.query_one("#query-input", Input).value
                if query:
                    self._trigger_search(query)

        self.push_screen(
            ConfirmModal(f"Remove '{Path(path).name}' from index?"),
            handle_confirm,
        )

    def action_filter_images(self) -> None:
        self.type_filter = "image"
        self._requery()

    def action_filter_text(self) -> None:
        self.type_filter = "text"
        self._requery()

    def action_filter_all(self) -> None:
        self.type_filter = None
        self._requery()

    def _requery(self) -> None:
        if self.mode == "search":
            query = self.query_one("#query-input", Input).value
            if query:
                self._trigger_search(query)
        elif self.mode == "list":
            self._load_list()

    def action_cycle_mode(self) -> None:
        idx = MODES.index(self.mode)
        self.mode = MODES[(idx + 1) % len(MODES)]

    # ── Reactive watches ──────────────────────────────────────────────────────

    def watch_mode(self, mode: str) -> None:
        self.sub_title = mode
        self.locked = False

        inp = self.query_one("#query-input", Input)
        lv = self.query_one("#results-list", ListView)
        status_display = self.query_one("#status-display", Static)

        if mode == "search":
            inp.display = True
            inp.placeholder = "search..."
            lv.display = True
            status_display.display = False
            inp.focus()

        elif mode == "list":
            inp.display = False
            lv.display = True
            status_display.display = False
            self._load_list()
            lv.focus()

        elif mode == "add":
            inp.display = True
            inp.placeholder = "path to index (press Enter)..."
            inp.value = ""
            lv.display = False
            status_display.display = False
            inp.focus()

        elif mode == "status":
            inp.display = False
            lv.display = False
            status_display.display = True
            self._refresh_status_display()

    def _load_list(self) -> None:
        if self._store is None:
            return
        items = self._store.list(type_filter=self.type_filter)
        lv = self.query_one("#results-list", ListView)
        lv.clear()
        for item in items:
            lv.append(FileItem(item.path, item.type))
        count = len(items)
        self._set_status(f"{count} item{'s' if count != 1 else ''} indexed")

    def _refresh_status_display(self) -> None:
        if self._store is None:
            return
        from glance.config import STORAGE_DIR
        counts = self._store.count()
        lines = [
            f"storage:  {STORAGE_DIR}",
            f"images:   {counts['image']}",
            f"text:     {counts['text']}",
            f"model:    clip-ViT-B-32",
        ]
        self.query_one("#status-display", Static).update("\n".join(lines))

    def _set_status(self, msg: str) -> None:
        self.query_one("#status-bar", Label).update(msg)


def launch() -> None:
    """Entry point called from cli.py when glance is run with no subcommand."""
    GlanceApp().run()
