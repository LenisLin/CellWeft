"""Cooperating cross-process readers and publishers on one shared filesystem."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import fcntl
import json


@contextmanager
def resource_lock(directory: Path, *, shared: bool = False, name: str = ".publication.lock") -> Iterator[None]:
    """Hold a shared read lease or exclusive write lease through file-handle closure."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def published_image(directory: Path, *, expected_request: dict,
                    required: tuple[str, ...]) -> Iterator[dict]:
    """Read current complete products while blocking replacement; keep handles in scope.

    Callers supply the dependency request obtained from current task inputs, not
    an old metadata snapshot. External readers must obey this lease protocol.
    """
    with resource_lock(directory, shared=True):
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("request") != expected_request:
            raise ValueError("Published image dependency revision is stale")
        if not set(required).issubset(metadata.get("completed", [])):
            raise ValueError("Requested image products are incomplete")
        if any(not Path(metadata['products'][key]).is_file() for key in required):
            raise ValueError("Requested image product is unavailable")
        yield metadata
