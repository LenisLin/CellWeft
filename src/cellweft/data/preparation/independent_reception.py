"""Read an existing independent study reception and its source relocation.

The original frozen scope, receipt and readback remain unchanged. Historical
owner paths are compared as declarations, while received products are checked
at their current NAS paths. This reader does not reopen retired owner data or
repeat scientific acceptance.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.inputs import file_revision
from cellweft.data.preparation.io_coordination import formal_read_lease
from cellweft.data.preparation.storage_return import _stable_digest
from cellweft.data.preparation.storage_return_study import _check_image_keys, _image_records


CONTINUATION_SCHEMA = "cellweft.independent_reception_source_relocation.v1"
EVIDENCE_SCHEMA = "cellweft.independent_reception_evidence_relocation.v1"
SUMMARY_SNAPSHOT_SCHEMA = "cellweft.independent_reception_summary_snapshot.v1"
MAX_CONTINUATIONS = 16
_BASIS = {"scope": "frozen_scope.json", "receipt": "receipt.json",
          "readback": "final_readback.json"}
_IDENTITY_FIELDS = ("device", "inode", "size", "mtime_ns")


def _read(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path != path.resolve() or not path.is_file():
        raise ValueError(f"Independent reception requires a canonical regular file: {path}")
    before = _identity(path)
    value = json.loads(path.read_text())
    if _identity(path) != before:
        raise ValueError(f"Independent reception document changed during read: {path}")
    if not isinstance(value, dict):
        raise ValueError(f"Expected a reception document: {path}")
    return value


def _identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return dict(zip(_IDENTITY_FIELDS, (value.st_dev, value.st_ino,
                                      value.st_size, value.st_mtime_ns)))


def _require_revision(expected: dict[str, Any]) -> None:
    if file_revision(Path(expected["path"])) != expected:
        raise ValueError(f"Independent reception version changed: {expected['path']}")


def scientific_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the summary fields preserved by a source-location-only change."""
    value = deepcopy(summary)
    value.pop("original_source")
    value["dependencies"].pop("raw")
    return value


def relocate_index_fields(summary: dict[str, Any], old: dict, new: dict) -> dict[str, Any]:
    """Change only the three current accepted-index reference locations."""
    value = deepcopy(summary)
    acceptance = value["acceptance"]
    declaration = acceptance["accepted_index"]
    if ({key: declaration[key] for key in ("path", "size", "mtime_ns")} != old
            or old["size"] != new["size"] or old["mtime_ns"] != new["mtime_ns"]
            or acceptance["technical_reception"] != old["path"]):
        raise ValueError("Independent accepted index binding differs")
    acceptance["technical_reception"] = new["path"]
    declaration["path"] = new["path"]
    matches = 0
    for qa in value["dependencies"]["qa"]:
        if "accepted_index" not in qa:
            continue
        if qa["accepted_index"] != {**declaration, "path": old["path"]}:
            raise ValueError("Independent QA accepted index binding differs")
        qa["accepted_index"]["path"] = new["path"]
        matches += 1
    if not matches:
        raise ValueError("Independent QA accepted index binding is missing")
    return value


def _continuations(path: Path, root: Path) -> list[tuple[Path, dict[str, Any], dict]]:
    """Read a bounded, exact-revision chain within this reception directory."""
    chain, seen = [], set()
    while True:
        if (not path.is_absolute() or path != path.resolve()
                or path.parent != root.resolve() or path in seen):
            raise ValueError("Independent continuation path or cycle is invalid")
        seen.add(path)
        revision = file_revision(path)
        record = _read(path)
        if file_revision(path) != revision:
            raise ValueError("Independent continuation changed during read")
        chain.append((path, record, revision))
        if record.get("schema") == CONTINUATION_SCHEMA:
            return list(reversed(chain))
        if record.get("schema") != EVIDENCE_SCHEMA or len(chain) >= MAX_CONTINUATIONS:
            raise ValueError("Independent continuation schema or chain bound differs")
        previous = record["predecessor_revision"]
        # Validate the path boundary before opening a claimed predecessor.
        path = Path(previous["path"])
        if not path.is_absolute() or path != path.resolve() or path.parent != root.resolve():
            raise ValueError("Independent predecessor escapes reception directory")
        _require_revision(previous)


def _check_index_move(move: dict, previous: dict, data_root: Path) -> None:
    old, new = move["old"], move["new"]
    declared = previous["scientific_summary"]["acceptance"]["accepted_index"]
    expected_parent = data_root / "processed/_shared/storage_reception/completed_studies/accepted_indexes"
    new_path = Path(new["path"])
    if (old != {key: declared[key] for key in ("path", "size", "mtime_ns")}
            or new_path != expected_parent / Path(old["path"]).name
            or new_path != new_path.resolve()
            or old["size"] != new["size"] or old["mtime_ns"] != new["mtime_ns"]
            or move["original_identity"] != move["received_identity"]
            or move["original_identity"]["inode"] != declared["inode"]
            or _identity(new_path) != move["received_identity"]):
        raise ValueError("Independent index relocation identity or path differs")
    _require_revision(new)


def _predecessor_summary(previous: dict, previous_revision: dict,
                         successor: dict, root: Path) -> dict:
    """An adopted snapshot binds the full body omitted by the old source schema."""
    if previous['schema'] == EVIDENCE_SCHEMA:
        return previous['summary_snapshot']
    expected = successor['predecessor_summary_snapshot_revision']
    if expected['path'] != str(root / 'source_summary_snapshot_20261110.json'):
        raise ValueError('Independent predecessor summary snapshot path differs')
    _require_revision(expected)
    snapshot = _read(Path(expected['path']))
    if (snapshot.get('schema') != SUMMARY_SNAPSHOT_SCHEMA
            or snapshot.get('study') != previous['study']
            or snapshot['predecessor_revision'] != previous_revision
            or snapshot['summary_revision'] != previous['summary_revision']
            or scientific_summary(snapshot['summary']) != previous['scientific_summary']
            or {key: snapshot['summary_identity'][key] for key in ('size', 'mtime_ns')}
            != {key: previous['summary_revision'][key] for key in ('size', 'mtime_ns')}):
        raise ValueError('Independent predecessor summary snapshot binding differs')
    _require_revision(expected)
    return snapshot['summary']


def _read_basis(study_root: Path, reception_root: Path) -> dict[str, Any]:
    documents = {key: _read(reception_root / name) for key, name in _BASIS.items()}
    scope, receipt, readback = (documents[key] for key in ("scope", "receipt", "readback"))
    study = study_root.name
    ids = sorted(item["image_id"] for item in scope["images"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Independent reception requires unique original image keys")
    if (scope["study"] != study or Path(scope["target"]).resolve() != study_root.resolve()
            or receipt["study"] != study or receipt["status"] != "complete"
            or receipt["received_count"] != len(ids)
            or readback["status"] != "passed"
            or readback["scope_images"] != len(ids)
            or readback["NAS_received_images"] != len(ids)
            or sorted(item["image_id"] for item in readback["images"]) != ids):
        raise ValueError("Independent reception documents disagree on the received scope")
    if Path(receipt["summary"]).resolve() != (study_root / "processing_summary.json").resolve():
        raise ValueError("Independent receipt names another summary")
    _require_revision(scope["images_table_revision"])
    table = _check_image_keys(study_root, study, ids)
    records = _image_records(study_root, study, ids, table)
    frozen = {item["image_id"]: item for item in scope["images"]}
    products, inputs, metadata = {}, {}, {}
    for record in records:
        image_id, value = record["image_id"], record["metadata"]
        accepted = frozen[image_id]
        storage = value["storage_consolidation"]
        # Source versions are historical evidence, not paths to reopen.
        if (storage["source_products"] != accepted["product_revisions"]
                or storage["source_input_identity"] != accepted["source_input_identity"]
                or _stable_digest(value["request"]) != accepted["request_digest"]):
            raise ValueError(f"Independent accepted binding changed: {image_id}")
        for expected in accepted["source_input_identity"]:
            _require_revision(expected)
        inputs[image_id] = accepted["source_input_identity"]
        products[image_id] = {name: file_revision(path)
                              for name, path in record["products"].items()}
        metadata[image_id] = file_revision(Path(record["path"]))
    tables = receipt["nas_five_table_revisions"]["after"]
    study_products = receipt["study_product_revisions"]["target"]
    for expected in (*tables.values(), *study_products.values()):
        _require_revision(expected)
    summary_path = study_root / "processing_summary.json"
    summary = _read(summary_path)
    if (summary["study"] != study or summary["scope"]["retained_images"] != len(ids)
            or summary["storage_consolidation"]["status"] != "complete"
            or summary["storage_consolidation"]["received_count"] != len(ids)):
        raise ValueError("Independent NAS summary does not retain the received scope")
    return {
        "study": study, "status": "received", "image_ids": ids,
        "basis": {key: file_revision(reception_root / name) for key, name in _BASIS.items()},
        "protected_versions": {"images": file_revision(study_root / "images.parquet"),
                               "inputs": inputs, "image_metadata": metadata,
                               "products": products, "tables": tables,
                               "study_products": study_products},
        "summary_revision": file_revision(summary_path),
        "scientific_summary": scientific_summary(summary), "summary": summary,
    }


def read_independent_reception(study_root: Path, *, data_root: Path,
                               reception_root: Path,
                               continuation_path: Path | None = None) -> dict[str, Any]:
    """Verify an original target-only reception and, when supplied, its continuation.

    ``reception_root`` contains the three original JSON documents. Omitting
    ``continuation_path`` reads their existing protocol for a migration baseline;
    supplying it additionally requires at most 16 immutable continuations rooted
    in the source-relocation record, the current summary and all retained raw
    object identities. Evidence successors change only registered index paths.
    Return exact current
    versions without reading pixel arrays. Missing files and version/scope
    conflicts fail; source/owner historical references are never reopened.
    """
    study_root, data_root, reception_root = map(Path, (study_root, data_root, reception_root))
    with formal_read_lease(study_root / "processing_summary.json", data_root=data_root):
        with resource_lock(study_root, shared=True):
            return _read_independent_reception(study_root, data_root=data_root,
                                               reception_root=reception_root,
                                               continuation_path=continuation_path)


def _read_independent_reception(study_root: Path, *, data_root: Path,
                                reception_root: Path,
                                continuation_path: Path | None) -> dict[str, Any]:
    """Read while the caller holds the formal lease and study publication lock."""
    snapshot = _read_basis(study_root, reception_root)
    if continuation_path is None:
        return snapshot
    chain = _continuations(Path(continuation_path), reception_root.parent)
    source_path, continuation, _ = chain[0]
    if (continuation["schema"] != CONTINUATION_SCHEMA
            or continuation["study"] != snapshot["study"]
            or continuation["status"] != "complete"
            or continuation["migration"] != "source_relocation"
            or continuation["original_receipt_summary_revision_bound"] is not False):
        raise ValueError("Invalid independent source-relocation continuation")
    for key in ("basis", "protected_versions"):
        if continuation[key] != snapshot[key]:
            raise ValueError(f"Independent continuation binding changed: {key}")
    previous, previous_revision = continuation, chain[0][2]
    snapshot_revisions = []
    for _, successor, successor_revision in chain[1:]:
        if (successor.get("study") != snapshot["study"]
                or successor.get("status") != "complete"
                or successor.get("migration") != "accepted_index_path_relocation"
                or successor["old_summary_revision"] != previous["summary_revision"]
                or any(successor[key] != snapshot[key] for key in ("basis", "protected_versions"))):
            raise ValueError("Independent evidence successor binding differs")
        _check_index_move(successor["index_move"], previous, data_root)
        full_before = _predecessor_summary(previous, previous_revision, successor,
                                           reception_root.parent)
        if previous['schema'] == CONTINUATION_SCHEMA:
            snapshot_revisions.append(successor['predecessor_summary_snapshot_revision'])
        expected_full = relocate_index_fields(full_before, successor['index_move']['old'],
                                             successor['index_move']['new'])
        if successor.get('summary_snapshot') != expected_full:
            raise ValueError('Independent evidence successor changes full summary')
        expected = relocate_index_fields(previous["scientific_summary"],
                                        successor["index_move"]["old"],
                                        successor["index_move"]["new"])
        if successor["scientific_summary"] != expected:
            raise ValueError("Independent evidence successor changes scientific summary")
        previous, previous_revision = successor, successor_revision
    for key in ("summary_revision", "scientific_summary"):
        if previous[key] != snapshot[key]:
            raise ValueError(f"Independent continuation binding changed: {key}")
    if len(chain) > 1 and previous['summary_snapshot'] != snapshot['summary']:
        raise ValueError('Independent evidence current full summary differs')
    raw = continuation["raw_source"]
    manifest_path = _check_raw_source(raw, snapshot["study"], data_root)
    source = snapshot["summary"]["original_source"]
    if (source["candidates"] != [str(manifest_path)]
            or snapshot["summary"]["dependencies"]["raw"] != [str(manifest_path)]
            or source["record"]["source_relocation_executed"] is not True
            or source["record"]["candidates"] != [str(manifest_path)]
            or source["record"]["evidence"] != str(source_path)):
        raise ValueError("Independent summary retains another active raw source")
    if any(file_revision(path) != revision for path, _, revision in chain):
        raise ValueError("Independent continuation changed during read")
    for revision in snapshot_revisions:
        _require_revision(revision)
    snapshot["continuation_revision"] = chain[-1][2]
    snapshot["raw_source"] = raw
    snapshot["continuation_chain"] = [revision for _, _, revision in chain]
    return snapshot


def _check_raw_source(raw: dict, study: str, data_root: Path) -> Path:
    manifest_path = Path(raw["manifest_revision"]["path"])
    raw_root = data_root / "raw/public" / study
    if manifest_path.resolve() != (raw_root / "manifest.json").resolve():
        raise ValueError("Independent continuation names another raw manifest")
    _require_revision(raw["manifest_revision"])
    if _identity(manifest_path) != raw["manifest_identity"]:
        raise ValueError("Independent raw manifest identity changed")
    manifest = _read(manifest_path)
    members = manifest["source_members"]
    if (manifest["study"] != study or len(members) != raw["members"]
            or sum(item["bytes"] for item in members) != raw["bytes"]):
        raise ValueError("Independent raw manifest scope changed")
    for member in members:
        path = Path(member["target"])
        if not path.resolve().is_relative_to(raw_root.resolve()):
            raise ValueError("Independent raw object escapes its study")
        if _identity(path) != member["target_identity"]:
            raise ValueError(f"Independent raw object changed: {path}")
    if _identity(manifest_path) != raw["manifest_identity"]:
        raise ValueError("Independent raw manifest changed during read")
    return manifest_path
