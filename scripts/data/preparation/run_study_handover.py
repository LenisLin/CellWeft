"""Transfer current accepted images to fixed owners; perform no downstream compute."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

from cellweft.data.preparation.delivery import build_delivery, upload_delivery
from cellweft.data.preparation.execution import Transport
from cellweft.data.preparation.inputs import file_revision
from cellweft.data.preparation.handover_versions import (delivery_version, new_pending,
    image_route_shard, image_scope_owners, next_study_cursor, packet_sources_current, pending_first, pending_packet_matches,
    packet_version, refresh_scope_receipt, resume_cursor, save_handover_state)
from cellweft.data.preparation.scheduling import _latest_d01_scopes
from cellweft.data.preparation.study_node import _claim
from cellweft.data.preparation.handover_priority import prefer_mask_updates


OLD_S1_REVISION = (451046, 1793406958444040277)
S1_FIELDS = ("Study", "Original marker", "Unified marker", "MAP-KG mapping",
             "UniProt accession", "Segmentation channel")


def _s1_rows_by_study(path: Path) -> dict[str, list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(S1_FIELDS):
            raise ValueError(f"S1 columns differ from the six-column contract: {path}")
        rows: dict[str, list[dict]] = {}
        for row in reader:
            rows.setdefault(row["Study"], []).append(row)
        return rows


def select_scoped_s1(canonical: Path, baseline: Path, study_cache_root: Path,
                     study: str | None, cache: dict,
                     *, old_revision: tuple[int, int] = OLD_S1_REVISION) -> Path:
    """Change a route's S1 packet version only when this study's rows change."""
    old = file_revision(baseline)
    if (old["size"], old["mtime_ns"]) != old_revision:
        raise ValueError(f"Route S1 baseline differs from the pinned version: {baseline}")
    if cache.get("baseline_revision") != old:
        cache["baseline_rows"] = _s1_rows_by_study(baseline)
        cache["baseline_revision"] = old
    current = file_revision(canonical)
    if cache.get("canonical_revision") != current:
        rows = _s1_rows_by_study(canonical)
        if file_revision(canonical) != current:
            raise ValueError("Canonical S1 changed while reading study rows")
        cache["canonical_rows"] = rows
        cache["canonical_revision"] = current
    if study is None:
        return baseline
    if Path(study).name != study or study in {".", ".."}:
        raise ValueError(f"Invalid S1 study name: {study}")
    current_rows = cache["canonical_rows"].get(study, [])
    study_copy = study_cache_root / f"{study}.csv"
    cached = cache.setdefault("study_caches", {})
    if study_copy.is_file():
        revision = file_revision(study_copy)
        if study not in cached or cached[study][0] != revision:
            cached[study] = (revision, _s1_rows_by_study(study_copy).get(study, []))
        if current_rows == cached[study][1]:
            return study_copy
    if current_rows == cache["baseline_rows"].get(study, []):
        return baseline
    study_cache_root.mkdir(parents=True, exist_ok=True)
    temporary = study_cache_root / f".{study}.csv.partial"
    shutil.copy2(canonical, temporary)
    if file_revision(canonical) != current:
        raise ValueError("Canonical S1 changed while caching study resource")
    os.replace(temporary, study_copy)
    cached[study] = (file_revision(study_copy), current_rows)
    return study_copy


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=["72", "73"], required=True)
    parser.add_argument("--owners", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
    parser.add_argument("--mask-bindings-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--gateway-host", required=True)
    parser.add_argument("--gateway-python", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True,
                        help="Existing checkpoint path on the target runtime")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-deliveries", type=int)
    parser.add_argument("--scope-burst", type=positive_int)
    parser.add_argument("--study-burst", type=positive_int,
                        help="maximum images delivered for one study in a feeder pass")
    parser.add_argument("--unknown-burst", type=positive_int)
    parser.add_argument("--prefer-mask-updates", action="store_true")
    parser.add_argument("--studies", type=Path, help="JSON list of this route's disjoint owned studies")
    parser.add_argument("--route-id", help="Separate source-backed staging and resource namespace")
    parser.add_argument("--shard-study", help="Partition this owned study across disjoint image routes")
    parser.add_argument("--shard-count", type=positive_int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--stage-queue-root", type=Path, help="Announce closed ready packets to CPU stages")
    parser.add_argument("--allow-unknown-mask-segmentation", action="store_true",
                        help="carry the explicit cpsam_v2 fallback policy in new packets")
    args = parser.parse_args()
    if not args.mask_bindings_root.is_dir():
        parser.error("--mask-bindings-root must be an existing directory")
    if args.max_deliveries is not None and args.max_deliveries < 0:
        parser.error("--max-deliveries must be nonnegative")
    scope_burst = args.scope_burst or (4 if args.node == "72" else 2)
    study_burst = args.study_burst or scope_burst
    owners = json.loads(args.owners.read_text())
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be within [0, shard-count)")
    if args.shard_count > 1 and not args.shard_study:
        parser.error("shard-study is required for image partitioning")
    if args.shard_study and owners.get(args.shard_study) != args.node:
        parser.error("shard-study must belong to this node")
    if args.route_id and (Path(args.route_id).name != args.route_id or args.route_id in {".", ".."}):
        parser.error("route-id must be one directory name")
    selected_studies = None
    if args.studies:
        selected_studies = json.loads(args.studies.read_text())
        if (not isinstance(selected_studies, list) or not selected_studies
                or any(not isinstance(name, str) for name in selected_studies)
                or len(set(selected_studies)) != len(selected_studies)
                or any(owners.get(name) != args.node for name in selected_studies)):
            parser.error("studies must be a nonempty unique list belonging to this owner")
    if args.shard_study and selected_studies is not None and args.shard_study not in selected_studies:
        parser.error("shard-study must be included in this route")
    root = args.node_root
    queue = root / "transfer/study_deliveries"
    gateway = {"id": args.node, "host": args.gateway_host,
               "python": str(args.gateway_python),
               "code_root": str(args.code_root)}
    weights = args.weights
    transport = Transport(gateway, queue)
    transport.lock_root = root / "locks"
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dir / f"handover_{args.node}.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"node": args.node, "scopes": {}, "errors": {}}
    partition = {"study": args.shard_study, "count": args.shard_count, "index": args.shard_index,
                 "assignment": "utf8_byte_sum_modulo"}
    if "route_partition" in state and state["route_partition"] != partition:
        parser.error("State image partition differs; migrate the drained route before changing ownership")
    state["route_partition"] = partition
    local_root = Path(".cache/cellweft/runtime")
    staging = local_root / "staging" / args.node
    if args.route_id:
        staging = staging / args.route_id
    staging.mkdir(parents=True, exist_ok=True)
    resources = local_root / "resources"
    sources = {
        "settings.yaml": Path("configs/data/preparation/defaults.yaml"),
        **{f"bank/{name}": args.data_root / "processed/_resources/protein_banks/map_gene_projector_v1" / name
           for name in ["manifest.json", "gene_symbols.json", "embeddings.pt"]},
    }
    # Each feeder owns a separate resource cache; source versions remain explicit.
    resources = resources / args.node
    if args.route_id:
        resources = resources / args.route_id
    canonical_s1 = args.data_root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
    scoped_s1_cache = {}
    def refresh_resources(study: str | None = None) -> Path:
        for relative, source in sources.items():
            target = resources / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = file_revision(source)
            if not target.is_file() or (target.stat().st_size, target.stat().st_mtime_ns) != (before["size"], before["mtime_ns"]):
                shutil.copy2(source, target)
                if file_revision(source) != before:
                    raise ValueError(f"Resource changed while caching: {source}")
        return select_scoped_s1(canonical_s1, resources / "s1.csv",
                                resources / "s1_by_study", study, scoped_s1_cache)

    refresh_resources()
    version_code = "import pathlib,json,sys;p=pathlib.Path(sys.argv[1]);s=p.stat();print(json.dumps({'path':str(p.resolve()),'size':s.st_size,'mtime_ns':s.st_mtime_ns}))"
    preferred = ["Lin2026_IMC_GastricCancer", "Keren2018_MIBI_TNBC"]
    studies = sorted((name for name, node in owners.items() if node == args.node
                      and (selected_studies is None or name in selected_studies)),
                     key=lambda name: (preferred.index(name) if name in preferred else len(preferred), name))
    headers = {}
    transferred = 0
    save_lock = threading.Lock()

    def save() -> None:
        with save_lock:
            state.update(pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
                         stop_requested=stop.is_set())
            save_handover_state(state_path, state)

    def build_heartbeat(done: threading.Event, errors: list[Exception]) -> None:
        while not done.wait(15):
            try:
                save()
            except Exception as exc:
                errors.append(exc)
                return

    with _claim(args.state_dir / f"handover_{args.node}.lock") as owned:
        if not owned:
            raise RuntimeError(f"A handover process already owns node {args.node}")
        state["cursor"] = resume_cursor(state, studies)
        save()
        if args.max_deliveries == 0:
            return
        while not stop.is_set():
            transferred_before_pass = transferred
            visit_cursor = state["cursor"]
            if args.prefer_mask_updates:
                resumed = resume_cursor(state, studies)
                if resumed.get("scope") and resumed.get("origin") in {"current", "pending"}:
                    visit_cursor = resumed
            first = studies.index(visit_cursor["study"])
            visit_order = studies[first:] + studies[:first]
            priority_errors = {}
            if args.prefer_mask_updates:
                masks_by_study = {}
                for study in visit_order:
                    if not any(key.startswith(f"{study}/") for key in state["scopes"]):
                        continue
                    mask_file = args.mask_bindings_root / f"{study}.json"
                    if mask_file.is_file():
                        try:
                            masks_by_study[study] = json.loads(mask_file.read_text())
                        except (OSError, ValueError) as exc:
                            priority_errors[study] = exc
                pending_study = visit_cursor["study"] if visit_cursor is not state["cursor"] else None
                visit_order = prefer_mask_updates(
                    visit_order, state["scopes"], masks_by_study, pending_study,
                    allow_unknown_mask_segmentation=args.allow_unknown_mask_segmentation)
            for study in visit_order:
                if stop.is_set():
                    break
                study_root = args.data_root / "processed" / study
                try:
                    if study in priority_errors:
                        raise priority_errors[study]
                    scopes, _ = _latest_d01_scopes(study_root)
                    scope_keys = [f"{study}/{scope['scope']}/{scope['record_id']}" for scope in scopes]
                    image_owners = image_scope_owners(scopes, scope_keys, state["scopes"])
                    scope_cursor = visit_cursor.get("scope") if visit_cursor["study"] == study else None
                    start_scope = scope_keys.index(scope_cursor) if scope_cursor in scope_keys else 0
                    pending_scopes = [index for index, key in enumerate(scope_keys)
                                      if any(image_owners.get(image) == key for image in
                                             state["scopes"].get(key, {}).get("pending", {}))]
                    if pending_scopes:
                        start_scope = min(start_scope, min(pending_scopes))
                    study_transferred = 0
                    for scope_index in range(start_scope, len(scopes)):
                        if stop.is_set() or study_transferred >= study_burst:
                            break
                        scope = scopes[scope_index]
                        key = scope_keys[scope_index]
                        try:
                            binding = scope["binding"]
                            receipt = state["scopes"].get(key)
                            if receipt is None or receipt["binding"] != binding:
                                receipt = refresh_scope_receipt(receipt, binding)
                                state["scopes"][key] = receipt
                            processed = set()
                            new_unknown_delivered = 0
                            for burst_index in range(scope_burst):
                                if stop.is_set() or study_transferred >= study_burst:
                                    break
                                current_scopes, _ = _latest_d01_scopes(study_root)
                                current_scope = next((item for item in current_scopes
                                                      if f"{study}/{item['scope']}/{item['record_id']}" == key), None)
                                if current_scope is None or current_scope["binding"] != binding:
                                    break
                                binding = current_scope["binding"]
                                revision = json.loads(transport.ssh(
                                    gateway, [gateway["python"], "-c", version_code, str(weights)],
                                    capture_output=True, text=True).stdout)
                                s1_path = refresh_resources(study)
                                mask_file = args.mask_bindings_root / f"{study}.json"
                                masks = json.loads(mask_file.read_text()) if mask_file.is_file() else {}
                                versions = receipt.setdefault("delivered_versions", {})
                                resource_versions = {}
                                for name in sources:
                                    version = file_revision(resources / name)
                                    resource_versions[f"resources/{name}"] = {key: version[key] for key in ("size", "mtime_ns")}
                                s1_version = file_revision(s1_path)
                                resource_versions["resources/s1.csv"] = {
                                    key: s1_version[key] for key in ("size", "mtime_ns")}
                                remaining = []
                                signatures = {}
                                for image in binding["image_ids"]:
                                    current = delivery_version(
                                        masks.get(image, {"status": "unknown"}),
                                        args.data_root, resource_versions, revision,
                                        allow_unknown_mask_segmentation=(
                                            args.allow_unknown_mask_segmentation))
                                    signatures[image] = current
                                    in_route = (study != args.shard_study or
                                                image_route_shard(image, args.shard_count) == args.shard_index)
                                    if in_route and image_owners[image] == key and image not in processed and (
                                        image not in receipt["delivered"] or versions.get(image) != current
                                    ):
                                        remaining.append(image)
                                if not remaining:
                                    break
                                if args.unknown_burst is not None and new_unknown_delivered >= args.unknown_burst:
                                    remaining = [image for image in remaining
                                                 if image in receipt["pending"] or
                                                 masks.get(image, {}).get("status") in {"bound", "absent", "unusable"}]
                                    if not remaining:
                                        break
                                if stop.is_set():
                                    break
                                image_id = pending_first(remaining, receipt["pending"], masks)
                                new_unknown = (image_id not in receipt["pending"] and
                                               masks.get(image_id, {}).get("status") not in
                                               {"bound", "absent", "unusable"})
                                state["cursor"] = {"study": study, "scope": key}
                                save()
                                pending = receipt["pending"].get(image_id)
                                if pending is not None:
                                    packet = Path(pending["local"])
                                    ready = packet / "ready.json"
                                    source_envelope = packet / "envelope.json"
                                    envelope_path = ready if ready.is_file() else source_envelope
                                    envelope = json.loads(envelope_path.read_text()) if envelope_path.is_file() else None
                                    if pending.get("version") is None and envelope is not None:
                                        pending["version"] = packet_version(envelope, image_id)
                                    stale = pending.get("version") != signatures[image_id]
                                    if envelope is not None:
                                        stale = not pending_packet_matches(
                                            pending, envelope, image_id, signatures[image_id],
                                            packet_sources_current(envelope, study_root=study_root,
                                                                   data_root=args.data_root))
                                        if envelope_path == source_envelope and not (packet / ".send-plan.json").is_file():
                                            stale = True
                                    if stale:
                                        if packet.exists():
                                            shutil.rmtree(packet)
                                        pending = None
                                if pending is None:
                                    if stop.is_set():
                                        break
                                    pending = new_pending(staging, queue, study, signatures[image_id])
                                    receipt["pending"][image_id] = pending
                                    save()
                                packet = Path(pending["local"])
                                pending["version"] = signatures[image_id]
                                if not (packet / "ready.json").is_file() and not (packet / "envelope.json").is_file():
                                    if packet.exists():
                                        shutil.rmtree(packet)
                                    if shutil.disk_usage(staging).free < 64 * 1024**3:
                                        raise OSError("Host transfer staging cannot preserve 64 GiB reserve")
                                    seeds = {}
                                    closed = Path("/tmp/cellweft_distributed_preparation") / study / image_id / "delivery.json"
                                    if closed.is_file():
                                        old = json.loads(closed.read_text())
                                        if old.get("stage") == "computed":
                                            seeds[image_id] = Path(old["metadata"])
                                    state["current"] = {"study": study, "image_id": image_id, "stage": "build", **pending}
                                    state["progress"] = {"stage": "build", "started_at": datetime.now(timezone.utc).isoformat()}
                                    save()
                                    build_done = threading.Event()
                                    heartbeat_errors: list[Exception] = []
                                    heartbeat = threading.Thread(target=build_heartbeat, args=(build_done, heartbeat_errors), daemon=True)
                                    heartbeat.start()
                                    build_started = time.monotonic()
                                    try:
                                        build_delivery(study_root, args.data_root, binding["image_ids"],
                                            {image_id: masks.get(image_id, {"status": "unknown"})},
                                            resources / "settings.yaml", weights, s1_path, resources / "bank",
                                            args.node, packet, revision, selected_image_ids=[image_id],
                                            seed_metadata=seeds, header_cache=headers, source_backed=True,
                                            allow_unknown_mask_segmentation=args.allow_unknown_mask_segmentation)
                                        pending["build_seconds"] = time.monotonic() - build_started
                                        if shutil.disk_usage(staging).free < 64 * 1024**3:
                                            raise OSError("Host transfer staging cannot preserve 64 GiB reserve")
                                    finally:
                                        build_done.set()
                                        heartbeat.join()
                                    if heartbeat_errors:
                                        raise heartbeat_errors[0]
                                state["current"] = {"study": study, "image_id": image_id, "stage": "transfer", **pending}
                                state["progress"] = {"stage": "transfer", "started_at": datetime.now(timezone.utc).isoformat()}
                                save()
                                previous = receipt["delivered"].get(image_id)
                                transfer_receipt = upload_delivery(
                                    packet, Path(pending["remote"]), transport, study_root, args.data_root,
                                    previous_delivery=Path(previous) if previous is not None else None)
                                if args.stage_queue_root is not None:
                                    command = ["env", f"PYTHONPATH={gateway['code_root']}/src", gateway["python"],
                                               "-c", "from pathlib import Path;import sys;from cellweft.data.preparation.stage_dispatch import announce_delivery;announce_delivery(Path(sys.argv[1]),queue_root=Path(sys.argv[2]),node_root=Path(sys.argv[3]))",
                                               pending["remote"], str(args.stage_queue_root), str(root)]
                                    transport.ssh(gateway, command, stdout=subprocess.DEVNULL)
                                transfer_receipt["build_seconds"] = pending.get("build_seconds")
                                receipt["delivered"][image_id] = pending["remote"]
                                receipt["delivered_versions"][image_id] = signatures[image_id]
                                processed.add(image_id)
                                if new_unknown:
                                    new_unknown_delivered += 1
                                study_transferred += 1
                                receipt.setdefault("transfer_receipts", {})[image_id] = transfer_receipt
                                del receipt["pending"][image_id]
                                state["errors"].pop(key, None)
                                state["current"] = None
                                state["progress"] = None
                                transferred += 1
                                state["cursor"] = ({"study": study, "scope": scope_keys[scope_index + 1]}
                                                   if scope_index + 1 < len(scopes)
                                                   else next_study_cursor(studies, study))
                                save()
                                shutil.rmtree(packet)
                                print(json.dumps({"node": args.node, "study": study, "image_id": image_id, "ready": pending["remote"]}), flush=True)
                                if args.max_deliveries is not None and transferred >= args.max_deliveries:
                                    stop.set()
                                    break
                        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                            message = f"{type(exc).__name__}: {exc}"
                            state["errors"][key] = message
                            save()
                            print(f"{key}: {message}", flush=True)
                            continue
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                    state["errors"][study] = f"{type(exc).__name__}: {exc}"
                    save()
                    print(f"{study}: {state['errors'][study]}", flush=True)
                if not stop.is_set():
                    state["cursor"] = next_study_cursor(studies, study)
                    save()
            save()
            if args.once:
                break
            if transferred == transferred_before_pass:
                stop.wait(15)


if __name__ == "__main__":
    main()
