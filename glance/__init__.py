import os
import sys
import multiprocessing.resource_tracker

# Disable multiprocessing in tokenizers and joblib/loky to avoid file descriptor conflicts
# when importing and using SentenceTransformers from background threads in Textual.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Monkeypatch multiprocessing.resource_tracker.ResourceTracker.ensure_running
# to temporarily wrap sys.stdout/sys.stderr. This prevents the tracker from
# calling .fileno() on redirected streams that return -1, which causes
# "ValueError: bad value(s) in fds_to_keep" on macOS.
_orig_ensure_running = multiprocessing.resource_tracker.ResourceTracker.ensure_running

def _safe_ensure_running(self):
    class SafeStream:
        def __init__(self, orig):
            self._orig = orig
        def fileno(self):
            # Raising OSError forces the resource tracker to safely skip appending
            # this stream's file descriptor to fds_to_pass.
            raise OSError("fileno disabled for resource tracker safety")
        def __getattr__(self, name):
            return getattr(self._orig, name)

    old_stderr = sys.stderr
    old_stdout = sys.stdout
    try:
        if sys.stderr is not None:
            sys.stderr = SafeStream(sys.stderr)
        if sys.stdout is not None:
            sys.stdout = SafeStream(sys.stdout)
        return _orig_ensure_running(self)
    finally:
        sys.stderr = old_stderr
        sys.stdout = old_stdout

multiprocessing.resource_tracker.ResourceTracker.ensure_running = _safe_ensure_running
