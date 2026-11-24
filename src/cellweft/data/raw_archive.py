"""Explicit, version-bound restoration of raw tar.gz members to local scratch.

This reader consumes an accepted complete member inventory. It neither creates
archives nor authorizes source retirement; existing format readers consume the
restored paths during the returned context's lifetime.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Iterator, Mapping, Sequence

from cellweft.data.preparation.capacity import SSD_RESERVE, CapacityReservations
from cellweft.data.preparation.io_coordination import formal_read_lease
from cellweft.data.raw_consolidation import file_identity


@dataclass(frozen=True)
class RawArchiveMember:
    """Original byte length and timestamp from the accepted archive inventory."""

    size: int
    mtime_ns: int


class _DigestingReader:
    """Hash encoded archive bytes as a gzip reader consumes them."""

    def __init__(self, stream):
        self._stream = stream
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        block = self._stream.read(size)
        self._digest.update(block)
        return block

    def drain(self) -> None:
        while self.read(4 * 1024**2):
            pass

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _resolved_data_root(data_root: Path) -> Path:
    try:
        root = Path(data_root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("data_root must resolve to an existing directory") from error
    if not root.is_dir():
        raise ValueError("data_root must resolve to an existing directory")
    return root


def _resolve_under(path: Path, root: Path, *, label: str, strict: bool) -> Path:
    try:
        resolved = Path(path).resolve(strict=strict)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{label} cannot be resolved") from error
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} must resolve inside data_root")
    return resolved


def _resolve_public_study_root(path: Path, data_root: Path) -> Path:
    if not Path(path).is_dir() or Path(path).is_symlink():
        raise ValueError("Study root must be an existing non-symlink directory")
    resolved = _resolve_under(path, data_root, label="Study root", strict=True)
    relative = resolved.relative_to(data_root)
    if len(relative.parts) != 3 or relative.parts[:2] != ("raw", "public"):
        raise ValueError("Study root must resolve to raw/public/<canonical_study> within data_root")
    return resolved


def _is_sha256_digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _member_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or not path.parts or path.is_absolute() or ".." in path.parts
            or str(path) != name or "\\" in name or "\0" in name):
        raise ValueError(f"Unsafe raw archive member: {name!r}")
    return name


def _check_identity(path: Path, expected: Mapping[str, int]) -> None:
    actual = file_identity(path)
    if any(actual[key] != expected[key] for key in ("device", "inode", "size", "mtime_ns")):
        raise ValueError(f"Raw archive version changed: {path}")


def _check_timestamp(entry: tarfile.TarInfo, expected_ns: int) -> None:
    try:
        valid = (entry.pax_headers.get("CELLWEFT.mtime_ns") == str(expected_ns)
                 and "mtime" in entry.pax_headers
                 and Decimal(entry.pax_headers["mtime"]) * 10**9 == expected_ns)
    except InvalidOperation:
        valid = False
    if not valid:
        raise ValueError(f"Raw archive member timestamp changed: {entry.name}")


@contextmanager
def restore_raw_members(
    archive: Path,
    *,
    archive_identity: Mapping[str, int],
    inventory: Mapping[str, RawArchiveMember],
    members: Sequence[str],
    scratch_root: Path,
    data_root: Path,
    archive_sha256: str,
    reserve_bytes: int = SSD_RESERVE,
    capacity_reservations: CapacityReservations | None = None,
) -> Iterator[dict[str, Path]]:
    """Restore exact accepted members, preserving bytes/mtime, then remove them.

    The archive uses names relative to its study directory, without a study
    wrapper. ``inventory`` covers every regular member, including manifest.json
    and metadata.json. Callers obtain the inventory, archive identity and
    ``archive_sha256`` from the accepted receipt, never by guessing paths or
    using legacy fallbacks.
    Scanning gzip is sequential even when only a subset is restored. When supplied
    by an accepted receipt, ``archive_sha256`` is checked in that same pass. No
    array decoding occurs. The NAS read lease covers scanning and identity checks;
    restored file consumers run against SSD after the lease is released.
    A supplied existing CapacityReservations manager holds this restore's growth
    through consumer completion. Without it, free-space checks are best effort;
    neither these checks nor the process-local manager reserve filesystem blocks
    against independent writers on the same SSD.
    """
    archive, scratch_root = map(Path, (archive, scratch_root))
    if not archive.name.endswith(".tar.gz"):
        raise ValueError("Expected a raw .tar.gz archive")
    resolved_data_root = _resolved_data_root(data_root)
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("Raw archive must be an existing regular non-symlink file")
    archive = _resolve_under(archive, resolved_data_root, label="Raw archive", strict=True)
    scratch_root = scratch_root.resolve()
    if scratch_root.is_relative_to(resolved_data_root):
        raise ValueError("Raw archive scratch must be outside the NAS data root")
    if not _is_sha256_digest(archive_sha256):
        raise ValueError("archive_sha256 must be a lowercase SHA-256 digest")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    for name, item in inventory.items():
        _member_name(name)
        if type(item.size) is not int or item.size < 0 or type(item.mtime_ns) is not int:
            raise ValueError(f"Invalid raw archive inventory entry: {name}")
    selected = set(members)
    if not selected or len(selected) != len(members) or not selected <= inventory.keys():
        raise ValueError("Select unique members from the complete accepted inventory")
    scratch_root.mkdir(parents=True, exist_ok=True)
    required = sum(inventory[name].size for name in selected)
    if shutil.disk_usage(scratch_root).free - required < reserve_bytes:
        raise OSError(f"Raw restore needs {required} bytes plus {reserve_bytes} reserve")
    with tempfile.TemporaryDirectory(prefix="raw_archive_", dir=scratch_root) as temporary:
        destination = Path(temporary)
        if capacity_reservations is not None:
            if not destination.resolve().is_relative_to(capacity_reservations.cache.resolve()):
                raise ValueError("Raw archive scratch must belong to the capacity manager's cache")
            capacity_reservations.acquire(destination, required)
        restored: dict[str, Path] = {}
        seen: set[str] = set()
        try:
            with formal_read_lease(archive, data_root=resolved_data_root):
                _check_identity(archive, archive_identity)
                with archive.open("rb") as compressed:
                    encoded_source = _DigestingReader(compressed)
                    with gzip.GzipFile(fileobj=encoded_source, mode="rb") as encoded:
                        with tarfile.open(fileobj=encoded, mode="r|") as source:
                            for entry in source:
                                name = _member_name(entry.name)
                                if entry.isdir():
                                    continue
                                if not entry.isfile() or entry.issparse():
                                    raise ValueError(f"Raw archive requires ordinary files: {name}")
                                if name in seen or name not in inventory:
                                    raise ValueError(f"Unexpected or duplicate raw archive member: {name}")
                                seen.add(name)
                                expected = inventory[name]
                                if entry.size != expected.size:
                                    raise ValueError(f"Raw archive member size changed: {name}")
                                _check_timestamp(entry, expected.mtime_ns)
                                if name not in selected:
                                    continue
                                target = destination / name
                                target.parent.mkdir(parents=True, exist_ok=True)
                                with source.extractfile(entry) as stream, target.open("xb") as output:
                                    while block := stream.read(4 * 1024**2):
                                        if shutil.disk_usage(scratch_root).free - len(block) < reserve_bytes:
                                            raise OSError("Raw restore would cross the SSD free-space reserve")
                                        output.write(block)
                                if target.stat().st_size != inventory[name].size:
                                    raise ValueError(f"Truncated raw archive member: {name}")
                                timestamp = inventory[name].mtime_ns
                                os.utime(target, ns=(timestamp, timestamp))
                                restored[name] = target
                        while encoded.read(4 * 1024**2):
                            pass
                    encoded_source.drain()
                    if encoded_source.hexdigest() != archive_sha256:
                        raise ValueError("Raw archive content digest changed")
                if seen != inventory.keys():
                    raise ValueError("Raw archive is missing accepted inventory members")
                _check_identity(archive, archive_identity)
            yield restored
        finally:
            if capacity_reservations is not None:
                capacity_reservations.release(destination)
