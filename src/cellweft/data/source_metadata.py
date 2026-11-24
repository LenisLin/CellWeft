"""One lossless JSON source-metadata file, with indexed member access.

JSON documents remain structured JSON. Text retains its exact UTF-8 content;
binary attachments retain their original bytes as base64. The byte index lets
readers fetch one document without parsing every study's verification records.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping
from urllib.parse import quote, unquote

from cellweft.data.raw_consolidation import file_identity
from cellweft.data.preparation.io_coordination import formal_read_lease, formal_write_lease


SCHEMA = "cellweft.source_metadata.v1"
_OFFSET = "00000000000000000000"
_STATIC = ("device", "inode", "size", "mtime_ns")


def _key(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"Invalid metadata member: {value}")
    return value


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant: {value}")


def _payload(path: Path, raw: bytes) -> tuple[str, bytes]:
    if path.suffix == ".json" and not raw.startswith(b"\xef\xbb\xbf"):
        try:
            json.loads(raw, parse_constant=_reject_constant)
            raw.decode("utf-8")
        except (ValueError, UnicodeError):
            pass  # Failed source responses are preserved as text/binary below.
        else:
            return "json", raw
    if path.suffix not in {".parquet", ".bin", ".range"} and b"\0" not in raw:
        try:
            return "utf8", _json(raw.decode("utf-8"))
        except UnicodeError:
            pass
    return "base64", _json(base64.b64encode(raw).decode("ascii"))


def read_metadata_index(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Read the owner and trailing index without loading document payloads."""
    with Path(path).open("rb") as stream:
        header = stream.read(4096).split(b',"documents":', 1)[0]
        owner = json.loads(header + b"}")
        if owner["schema"] != SCHEMA or not re.fullmatch(r"\d{20}", owner["index_offset"]):
            raise ValueError(f"Unknown source-metadata format: {path}")
        stream.seek(int(owner["index_offset"]))
        text = stream.read().decode("utf-8")
        index, end = json.JSONDecoder().raw_decode(text)
        if text[end:].strip() != "}":
            raise ValueError(f"Invalid source-metadata footer: {path}")
    for key in index:
        _key(key)
    return owner["owner"], index


def read_metadata_member(path: Path, key: str, *, index: Mapping[str, dict[str, Any]] | None = None) -> bytes:
    """Recover one member's original bytes, including JSON whitespace."""
    entries = read_metadata_index(path)[1] if index is None else index
    item = entries[_key(key)]
    with Path(path).open("rb") as stream:
        stream.seek(item["offset"])
        payload = stream.read(item["length"])
    if len(payload) != item["length"]:
        raise ValueError(f"Truncated source-metadata member: {path}#{key}")
    if item["encoding"] == "json":
        raw = payload
    elif item["encoding"] == "utf8":
        raw = json.loads(payload).encode("utf-8")
    elif item["encoding"] == "base64":
        raw = base64.b64decode(json.loads(payload), validate=True)
    else:
        raise ValueError(f"Unknown metadata encoding: {item['encoding']}")
    if len(raw) != item["source_identity"]["size"]:
        raise ValueError(f"Metadata source length mismatch: {path}#{key}")
    return raw


def build_metadata_bundle(path: Path, sources: Mapping[str, Path], *, owner: Mapping[str, Any],
                          expected: Mapping[str, dict[str, int]] | None = None) -> dict[str, Any]:
    """Atomically write and byte-verify a bundle; never remove source files."""
    path = Path(path)
    identities = {key: file_identity(source) for key, source in sources.items()}
    for key, identity in identities.items():
        _key(key)
        if expected is not None and any(identity[k] != expected[key][k] for k in _STATIC):
            raise ValueError(f"Metadata source changed before consolidation: {sources[key]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    header = _json({"schema": SCHEMA, "index_offset": _OFFSET, "owner": dict(owner)})
    offset_position = header.index(_OFFSET.encode())
    index: dict[str, dict[str, Any]] = {}
    small_duplicates: dict[bytes, str] = {}
    duplicates = 0
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(header[:-1] + b',"documents":{')
            first = True
            for key, source in sources.items():
                raw = Path(source).read_bytes()
                identity = file_identity(source)
                if any(identity[k] != identities[key][k] for k in _STATIC):
                    raise ValueError(f"Metadata source changed while reading: {source}")
                original = small_duplicates.get(raw) if len(raw) <= 65536 else None
                if original is not None:
                    item = {**index[original], "source_identity": identity, "same_content_as": original}
                    duplicates += 1
                else:
                    encoding, payload = _payload(Path(source), raw)
                    stream.write((b"" if first else b",") + _json(key) + b':{"encoding":' + _json(encoding) + b',"value":')
                    first = False
                    item = {"offset": stream.tell(), "length": len(payload),
                            "encoding": encoding, "source_identity": identity}
                    stream.write(payload + b"}")
                    if len(raw) <= 65536:
                        small_duplicates[raw] = key
                index[key] = item
            stream.write(b'},"index":')
            index_offset = stream.tell()
            stream.write(_json(index) + b"}\n")
            stream.seek(offset_position)
            stream.write(f"{index_offset:020d}".encode())
            stream.flush()
            os.fsync(stream.fileno())
        _, actual_index = read_metadata_index(temporary)
        for key, source in sources.items():
            identity = file_identity(source)
            if any(identity[k] != identities[key][k] for k in _STATIC):
                raise ValueError(f"Metadata source changed before verification: {source}")
            if read_metadata_member(temporary, key, index=actual_index) != Path(source).read_bytes():
                raise ValueError(f"Metadata byte readback failed: {source}")
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"members": len(index), "deduplicated_payloads": duplicates,
            "source_bytes": sum(x["size"] for x in identities.values()), "bundle_bytes": path.stat().st_size}


@dataclass
class _Materialized:
    directory: tempfile.TemporaryDirectory
    bundle: Path
    data_root: Path
    prefix: str
    identity: dict[str, int]


_MATERIALIZED: dict[tuple[Path, str], _Materialized] = {}


def metadata_source_directory(data_root: Path, study: str, *, scope: str = "public") -> Path:
    """Stage a source namespace on SSD for existing file-based scientific readers.

    The temporary directory is removed at interpreter exit. There is no legacy
    NAS-path fallback. Original member bytes and mtimes are restored exactly.
    """
    if scope not in {"public", "private", "raw", "shared"} or not study or "/" in study:
        raise ValueError("Expected a source-metadata scope and one study identifier")
    data_root = Path(data_root)
    bundle = (data_root / "raw/shared/metadata.json" if scope == "shared" else
              data_root / "raw" / ("public" if scope == "raw" else scope) / study / "metadata.json")
    prefix = "shared/" if scope == "shared" else f"{scope}/{study}/"
    cache_key = (bundle, prefix)
    if cache_key in _MATERIALIZED:
        return Path(_MATERIALIZED[cache_key].directory.name)
    scratch = Path(os.environ.get("CELLWEFT_METADATA_SCRATCH", Path(__file__).resolve().parents[3] / "cache/source_metadata_readers"))
    scratch.mkdir(parents=True, exist_ok=True)
    directory = tempfile.TemporaryDirectory(prefix=f"{study}_", dir=scratch)
    destination = Path(directory.name)
    try:
        with formal_read_lease(bundle, data_root=data_root):
            _, index = read_metadata_index(bundle)
            members = [key for key in index if key.startswith(prefix)]
            if not members:
                raise KeyError(f"No source namespace {prefix} in {bundle}")
            identity = file_identity(bundle)
            for key in members:
                target = destination / key.removeprefix(prefix)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(read_metadata_member(bundle, key, index=index))
                timestamp = index[key]["source_identity"]["mtime_ns"]
                os.utime(target, ns=(timestamp, timestamp))
        _MATERIALIZED[cache_key] = _Materialized(directory, bundle, data_root, prefix, identity)
    except BaseException:
        directory.cleanup()
        raise
    return destination


def metadata_source_relative(path: Path, root: Path) -> Path:
    """Return stable bundle/member provenance for staged metadata paths."""
    path, root = Path(path), Path(root)
    for item in _MATERIALIZED.values():
        if root != item.data_root:
            continue
        try:
            relative = path.relative_to(item.directory.name)
        except ValueError:
            continue
        member = item.prefix + (relative.as_posix() if str(relative) != "." else "")
        return Path(item.bundle.relative_to(root).as_posix() + "#" + quote(member, safe="/"))
    return path.relative_to(root)


def metadata_reference_path(data_root: Path, reference: str) -> Path:
    """Resolve a canonical metadata.json#member reference to an SSD file."""
    bundle_name, separator, member = reference.partition("#")
    if not separator:
        raise ValueError("Metadata references require a bundle and member fragment")
    bundle = Path(data_root) / bundle_name
    member = unquote(member)
    parts = PurePosixPath(_key(member)).parts
    if parts[0] == "shared":
        if bundle != Path(data_root) / "raw/shared/metadata.json":
            raise ValueError(f"Shared metadata owner does not match member: {reference}")
        return metadata_source_directory(Path(data_root), "_shared", scope="shared").joinpath(*parts[1:])
    directory = metadata_source_directory(Path(data_root), parts[1], scope=parts[0])
    expected = Path(data_root) / "raw" / ("public" if parts[0] == "raw" else parts[0]) / parts[1] / "metadata.json"
    if bundle != expected:
        raise ValueError(f"Metadata owner does not match member: {reference}")
    return directory.joinpath(*parts[2:])


def publish_metadata_sources(directory: Path) -> None:
    """Publish explicit scientific source-followup edits from an SSD namespace."""
    directory = Path(directory)
    matches = [item for item in _MATERIALIZED.values() if directory.is_relative_to(item.directory.name)]
    if len(matches) != 1:
        raise ValueError(f"Not one staged metadata namespace: {directory}")
    item = matches[0]
    with formal_write_lease(item.bundle, data_root=item.data_root):
        current = file_identity(item.bundle)
        if any(current[k] != item.identity[k] for k in _STATIC):
            raise ValueError(f"Source metadata changed since staging: {item.bundle}")
        owner, index = read_metadata_index(item.bundle)
        with tempfile.TemporaryDirectory(dir=Path(item.directory.name).parent) as scratch:
            sources = {}
            for key, entry in index.items():
                source = Path(scratch) / key
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(read_metadata_member(item.bundle, key, index=index))
                stamp = entry["source_identity"]["mtime_ns"]
                os.utime(source, ns=(stamp, stamp))
                sources[key] = source
            for source in Path(item.directory.name).rglob("*"):
                if source.is_file() and not source.name.endswith(".lock"):
                    key = item.prefix + source.relative_to(item.directory.name).as_posix()
                    sources[key] = source
            build_metadata_bundle(item.bundle, sources, owner=owner)
        item.identity = file_identity(item.bundle)
