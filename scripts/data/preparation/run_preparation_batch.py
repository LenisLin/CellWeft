"""Dynamically admit current D01 scopes to the shared preparation controller.

The batch entry point owns discovery and the controller lease.  Scientific
approval remains in the reviewed study tables; discovery only materializes
current bindings and reports scopes that cannot be consumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import time


def _run_controller(run_distributed, plan: dict, discover_callback):
    """Call old finite controllers and the continuous controller through one API."""
    parameters = inspect.signature(run_distributed).parameters
    accepts_discover = "discover" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_discover:
        return run_distributed(plan, discover=discover_callback)
    return run_distributed(plan)


def _eligibility(studies: list[dict], gaps: list[dict]) -> dict:
    scopes = []
    for entry in studies:
        scopes.append({
            "study": entry["study"],
            "scope": entry.get("scope"),
            "record_id": entry.get("record_id"),
            "scope_key": entry.get("scope_key"),
            "images": list(entry.get("image_ids", [])),
            "known_masks": sum(
                value.get("status") != "unknown"
                for value in entry.get("mask_bindings", {}).values()
            ),
        })
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "accepted_scopes": scopes,
        "accepted_studies": [
            {"study": entry["study"], "images": len(entry.get("image_ids", []))}
            for entry in studies
        ],
        "gaps": gaps,
        "input_approvals_modified": False,
    }


def main() -> None:
    """Wait for the deployment time, discover current scopes, then run once or continuously."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "4" if key == "OMP_NUM_THREADS" else "1"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--continuous", action="store_true",
                        help="keep the controller alive and re-discover accepted scopes")
    parser.add_argument("--poll-seconds", type=float,
                        help="override the configured dynamic discovery interval")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.continuous:
        config["continuous"] = True
    if args.poll_seconds is not None:
        if args.poll_seconds <= 0:
            raise ValueError("--poll-seconds must be positive")
        config["discover_interval_seconds"] = args.poll_seconds
    if "poll_seconds" not in config:
        config["poll_seconds"] = float(config.get("discover_interval_seconds", 30))

    due_value = config.get("not_before")
    if due_value:
        due = datetime.fromisoformat(due_value)
        if due.tzinfo is None:
            due = due.replace(tzinfo=timezone.utc)
        while datetime.now(timezone.utc) < due:
            time.sleep(max(0, min(30, (due - datetime.now(timezone.utc)).total_seconds())))

    from cellweft.data.io.publication import resource_lock
    from cellweft.data.io.writers import write_json
    from cellweft.data.preparation.execution import run_distributed
    from cellweft.data.preparation.scheduling import discover as discover_scopes

    root = Path(config["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    continuous = bool(config.get("continuous", False))
    with resource_lock(Path(config["nas_lock_root"]), name="distributed_controller.lock"):
        status_path=root/'status.json'
        previous=json.loads(status_path.read_text()) if status_path.is_file() else {}
        write_json(status_path,{**previous,'finished':False,'controller_stage':'discovering_current_D01',
                                'scope_currently_revalidated':False,'updated_at':time.time()})
        # Continuous execution starts with an empty snapshot.  The controller's
        # background discovery callback admits scopes as each one is validated;
        # doing a synchronous full scan here would block already-ready images.
        if continuous:
            studies, gaps = [], []
        else:
            studies, gaps = discover_scopes(config)
        write_json(root / "eligibility.json", _eligibility(studies, gaps))
        plan = {**config, "studies": studies, "discovery_gaps": gaps}
        write_json(root / "plan.json", plan)

        if not studies and not continuous:
            write_json(root / "status.json", {
                "finished": True,
                "total": 0,
                "outcomes": [],
                "status": "no_eligible_input_scope",
                "gaps": gaps,
            })
            return

        acceptance_path = config.get("deployment_acceptance")
        if acceptance_path:
            receipt = Path(acceptance_path)
            while not receipt.is_file() or not json.loads(receipt.read_text()).get("passed"):
                write_json(root / "status.json", {
                    "finished": False,
                    "total": len(studies),
                    "status": "waiting_for_deployment_validation",
                    "outcomes": [],
                    "gaps": gaps,
                })
                time.sleep(30)

        def discover_current(on_scope=None):
            current, current_gaps = discover_scopes(config, on_scope=on_scope)
            write_json(root / "eligibility.json", _eligibility(current, current_gaps))
            write_json(root / "plan.json", {**config, "studies": current,
                                           "discovery_gaps": current_gaps})
            return current, current_gaps

        _run_controller(run_distributed, plan, discover_current if continuous else None)


if __name__ == "__main__":
    main()
