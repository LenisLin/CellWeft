"""Deliver one accepted study scope to its fixed 72/data or 73/public owner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import uuid

import pandas as pd

from cellweft.data.preparation.delivery import build_delivery, upload_delivery
from cellweft.data.preparation.execution import Transport




def ready_studies(root: Path, limit: int) -> list[Path]:
    """Return a bounded directory scan; D01 is checked during each build."""
    return sorted(path for path in root.iterdir() if path.is_dir() and
                  (path / "review_results.csv").is_file())[:limit]


def accepted_scopes(study_root: Path) -> list[list[str]]:
    """Discover explicit current D01 scopes; the builder still verifies each binding."""
    rows = pd.read_csv(study_root / "review_results.csv", keep_default_na=False)
    latest = rows.drop_duplicates(["scope", "record_id", "check_id"], keep="last")
    scopes = set()
    for row in latest[(latest.check_id == "D01") & (latest.result == "pass")].itertuples():
        try:
            ids = json.loads(row.details)["input_binding"]["image_ids"]
        except (ValueError, KeyError, TypeError):
            continue
        if ids:
            scopes.add(tuple(sorted(ids)))
    return [list(ids) for ids in sorted(scopes)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--study-root", type=Path)
    scope.add_argument("--scan-root", type=Path)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--image-id", action="append")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-bindings-root", type=Path, required=True)
    parser.add_argument("--owner-map", type=Path, required=True,
                        help="JSON study -> node ID; fixed owner for every delivery")
    parser.add_argument("--node", choices=("72", "73"), required=True)
    parser.add_argument("--node-root", type=Path, required=True,
                        help="Explicit target project root for the logical owner")
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True,
                        help="Existing target runtime checkpoint path")
    parser.add_argument("--gateway-host", required=True)
    parser.add_argument("--gateway-python", type=Path, required=True)
    parser.add_argument("--gateway-code-root", type=Path, required=True)
    parser.add_argument("--lock-root", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1 or args.scan_root and args.image_id:
        parser.error("Scan limit must be positive; image selection requires one study")
    owners = json.loads(args.owner_map.read_text())
    if args.study_root:
        selected = args.image_id or (accepted_scopes(args.study_root)[0]
                                     if accepted_scopes(args.study_root) else None)
        if selected is None:
            raise ValueError("No explicit current D01 image scope for study")
        jobs = [(args.study_root, selected)]
    else:
        jobs = [(study, ids) for study in ready_studies(args.scan_root, args.limit)
                for ids in accepted_scopes(study)][:args.limit]
    target = args.node_root / "transfer" / "study_deliveries"
    gateway = {"id": args.node, "host": args.gateway_host,
               "python": str(args.gateway_python), "code_root": str(args.gateway_code_root)}
    transport = Transport(gateway, target)
    transport.lock_root = args.lock_root
    weight_code = ("import json,pathlib,sys; p=pathlib.Path(sys.argv[1]);s=p.stat();"
                   "print(json.dumps({'path':str(p.resolve()),'size':s.st_size,'mtime_ns':s.st_mtime_ns}))")
    weight_result = transport.ssh(gateway, [str(args.gateway_python), "-c", weight_code,
        str(args.weights)], capture_output=True, text=True)
    weights_revision = json.loads(weight_result.stdout)
    for study_root, ids in jobs:
        study = study_root.name
        if owners.get(study) != args.node:
            raise ValueError(f"{study}: fixed owner differs from selected node")
        bindings = json.loads((args.mask_bindings_root / f"{study}.json").read_text())
        delivery_id = uuid.uuid4().hex
        destination = target / study / delivery_id
        with tempfile.TemporaryDirectory(prefix="cellweft_delivery_") as temporary:
            packet = Path(temporary) / delivery_id
            build_delivery(study_root, args.data_root, ids,
                           {image_id: bindings[image_id] for image_id in ids},
                           args.settings, args.weights, args.s1, args.bank_dir,
                           args.node, packet, weights_revision)
            upload_delivery(packet, destination, transport, study_root, args.data_root)
        print(json.dumps({"study": study, "node": args.node,
                          "delivery": str(destination), "image_ids": ids}))


if __name__ == "__main__":
    main()
