"""Load the storage-return runner from its persisted processed reception."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import stat
from types import ModuleType
from uuid import uuid4

from cellweft.data.preparation.storage_reception_relocation import (
    activation_record, storage_return_runner_lock,
)


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _require_directory(path: Path, label: str) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(mode):
        raise RuntimeError(f"{label} is not a directory: {path}")
    return True


def select_storage_return_roots(project: Path, nas: Path
                                ) -> tuple[Path, Path, str]:
    """Select only the activated processed reception; legacy roots are retired."""
    project, nas = Path(project), Path(nas)
    if not _require_directory(nas, "NAS data root"):
        raise RuntimeError(f"NAS data root is unavailable: {nas}")
    reception = nas / "processed/_shared/storage_reception/completed_studies"
    if not _require_directory(reception, "processed storage-reception root"):
        raise RuntimeError(f"processed storage-reception root is unavailable: {reception}")

    fence = reception / "legacy_root_retirement.json"
    activation = reception / "relocation_activation.json"
    has_fence, has_activation = _path_exists(fence), _path_exists(activation)
    if not has_fence and not has_activation:
        raise RuntimeError("persisted processed storage-reception activation is required")
    if has_fence != has_activation:
        raise RuntimeError("reception activation is incomplete")

    try:
        record, _, _ = activation_record(reception)
        if not isinstance(record.get("studies"), dict) or not record["studies"]:
            raise ValueError("activation roster is missing")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("persisted processed storage-reception activation is invalid") from error
    return reception / "execution", reception, "activated"


def load_storage_return_runner(path: Path) -> ModuleType:
    """Fresh-import a reader bound to the activated processed reception."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        f"cellweft_storage_return_{uuid4().hex}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load storage-return runner: {path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner
