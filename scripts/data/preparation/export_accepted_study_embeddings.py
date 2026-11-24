"""Export one coherent marker-vector asset for each currently accepted study.

The image controller and this vector controller share D01 discovery, but a
study with multiple accepted scopes receives one vector task whose panel set
is the union of those exact scopes.  Each scope is revalidated independently
before publication.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time


def _task_key(task: dict) -> str:
    """Return a stable identity for the accepted scope and vector inputs."""
    return json.dumps(task, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _vector_task_key(task: dict, *, s1_path: Path, bank_dir: Path) -> str:
    """Key cache entries by scientific bindings and actual S1/bank revisions."""
    from cellweft.data.preparation.inputs import file_revision, processing_binding

    scientific = {**task, "scopes": []}
    for scope in task.get("scopes", []):
        current = dict(scope)
        binding = current.get("input_binding")
        if binding is not None:
            current["input_binding"] = processing_binding(binding)
            # The discovered scope key includes sample/clinical revisions;
            # use the projected scientific binding so metadata-only rebinds
            # can reuse the existing vector asset.
            current["scope_key"] = _task_key(current["input_binding"])
        scientific["scopes"].append(current)
    assets = {
        "s1": file_revision(s1_path),
        "bank_embeddings": file_revision(bank_dir / "embeddings.pt"),
        "bank_manifest": file_revision(bank_dir / "manifest.json"),
        "bank_genes": file_revision(bank_dir / "gene_symbols.json"),
    }
    return _task_key({"task": scientific, "assets": assets})


def _published_current(study_root: Path, study: str, locks: Path) -> bool:
    """Check a completed vector asset while consuming one shared read slot."""
    from cellweft.data.preparation.embeddings import published_study_embeddings
    from cellweft.data.preparation.remote_worker import limited_slots

    try:
        with limited_slots(locks, "nas_read", 2):
            with published_study_embeddings(study_root, expected_study=study):
                return True
    except (OSError, KeyError, ValueError, RuntimeError):
        return False


def _write_status(path: Path, *, outcomes: dict[str, dict], gaps: list[dict],
                  continuous: bool, finished: bool, status: str) -> None:
    from cellweft.data.io.writers import write_json

    write_json(path, {
        "outcomes": list(outcomes.values()),
        "gaps": gaps,
        "continuous": continuous,
        "finished": finished,
        "status": status,
    })


def _discover(config: dict, cache: dict, *, on_scope=None, on_study=None,
              on_progress=None) -> tuple[list[dict], list[dict]]:
    from cellweft.data.preparation.scheduling import discover, save_header_cache

    studies, gaps = discover(
        config,
        header_cache=cache,
        on_scope=on_scope,
        on_study=on_study,
        on_progress=on_progress,
    )
    save_header_cache(config, cache)
    return studies, gaps


def main() -> None:
    """Consume the current accepted scopes once or keep listening for new ones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--continuous", action="store_true",
                        help="re-discover accepted scopes after each polling interval")
    parser.add_argument("--poll-seconds", type=float,
                        help="override discover_interval_seconds from the plan")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    continuous = bool(plan.get("continuous", False) or args.continuous)
    poll_seconds = args.poll_seconds
    if poll_seconds is None:
        poll_seconds = float(plan.get("discover_interval_seconds", 60))
    if poll_seconds <= 0:
        raise ValueError("poll interval must be positive")

    import torch
    torch.set_num_threads(2)
    from cellweft.data.io.publication import resource_lock
    from cellweft.data.io.writers import write_json
    from cellweft.data.preparation.embeddings import export_study_embedding_task
    from cellweft.data.preparation.scheduling import load_header_cache, vector_tasks
    from cellweft.data.preparation.local_resources import admission_paused

    data_root = Path(plan["data_root"])
    run_root = Path(plan["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    def draining() -> bool:
        return stop.is_set() or (run_root/'STOP').exists() or (run_root/'STOP_ADMISSION').exists()
    from cellweft.data.preparation.io_coordination import default_io_root
    locks = Path(plan.get("nas_lock_root", default_io_root(data_root)))
    cache = load_header_cache(plan)
    status_path = run_root / "embedding_status.json"
    outcomes: dict[str, dict] = {}
    gaps: list[dict] = []
    if status_path.is_file():
        previous_status = json.loads(status_path.read_text())
        for outcome in previous_status.get("outcomes", []):
            if isinstance(outcome, dict) and outcome.get("study"):
                outcomes[str(outcome["study"])] = outcome
        if isinstance(previous_status.get("gaps"), list):
            gaps = list(previous_status["gaps"])

    def process_task(task: dict) -> None:
        """Consume one study after all of its exact scopes are discovered."""
        if draining():return
        while admission_paused(plan):
            if draining():return
            stop.wait(5)
        study = task["study"]
        key = _vector_task_key(task, s1_path=args.s1, bank_dir=args.bank)
        previous = outcomes.get(study)
        if previous and previous.get("status") == "complete" and previous.get("task_key") == key:
            if _published_current(data_root / "processed" / study, study, locks):
                outcomes[study] = {
                    **previous,
                    "image_ids": list(task.get("image_ids", [])),
                    "scopes": [scope.get("scope_key") for scope in task.get("scopes", [])],
                    "panel_ids": list(task.get("panel_ids", [])),
                    "reused": True,
                }
                return
        try:
            result = export_study_embedding_task(
                task,
                data_root=data_root,
                s1_path=args.s1,
                bank_dir=args.bank,
                nas_lock_root=locks,
                header_cache=cache,
            )
            outcomes[study] = {**result, "task_key": key, "reused": False}
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            outcomes[study] = {
                "study": study,
                "status": "gap",
                "task_key": key,
                "error": str(error),
            }
        _write_status(status_path, outcomes=outcomes, gaps=gaps,
                      continuous=continuous, finished=False, status="running")

    def process_once() -> bool:
        nonlocal gaps, plan
        _write_status(status_path, outcomes=outcomes, gaps=gaps,
                      continuous=continuous, finished=False, status='running')
        incremental = set()

        def on_study(entries: list[dict]) -> None:
            # discover calls this after every study's scopes have been
            # validated.  One task per callback preserves the study panel
            # union while allowing earlier studies to publish immediately.
            for task in vector_tasks(entries):
                key = _vector_task_key(task, s1_path=args.s1, bank_dir=args.bank)
                incremental.add(key)
                process_task(task)

        studies, discovery_gaps = _discover(
            plan,
            cache,
            on_study=on_study if continuous else None,
        )
        if draining():return False
        tasks = vector_tasks(studies)
        gaps = list(discovery_gaps)
        plan = {**plan, "studies": studies, "discovery_gaps": gaps}
        write_json(run_root / "plan.json", plan)
        current_studies = {task["study"] for task in tasks}
        for study, outcome in list(outcomes.items()):
            if study not in current_studies and outcome.get("status") == "complete":
                outcomes[study] = {**outcome, "status": "withdrawn"}

        for task in tasks:
            key = _vector_task_key(task, s1_path=args.s1, bank_dir=args.bank)
            if key in incremental:
                continue
            process_task(task)
        _write_status(
            status_path,
            outcomes=outcomes,
            gaps=gaps,
            continuous=continuous,
            finished=not continuous,
            status=(
                "waiting_for_accepted_scope" if continuous and not tasks else
                "no_accepted_scope" if not tasks else
                "gap" if any(outcome.get("status") == "gap"
                             for outcome in outcomes.values()) else
                "complete"
            ),
        )
        return bool(tasks)

    with resource_lock(locks, name="embedding_controller.lock"):
        while not draining():
            process_once()
            if not continuous:
                break
            stop.wait(poll_seconds)
        if draining():
            _write_status(status_path, outcomes=outcomes, gaps=gaps,
                          continuous=continuous, finished=True, status='drained')


if __name__ == "__main__":
    main()
