"""Poll immutable ready studies with one independent CPU embedding process."""
from __future__ import annotations

import argparse
import json
import os
# Set BLAS limits before importing numerical modules.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
import signal
import sys
import threading
from pathlib import Path

from cellweft.data.io.writers import write_json
from cellweft.data.preparation.study_node import NODE_CONFIG, _claim, process_study_embeddings

RECOVERABLE = (OSError, ValueError, KeyError, RuntimeError)


def run_embeddings(*, node_id: str, queue_dir: Path, node_root: Path,
                   once: bool, poll_seconds: float, stopped: threading.Event,
                   study: str | None = None) -> dict:
    """Close admitted study tasks before stopping; retry only unsuccessful work."""
    if node_id not in NODE_CONFIG or poll_seconds <= 0:
        raise ValueError("Node must be 72/73 and poll interval must be positive")
    if study is not None and (Path(study).name != study or study in {".", ".."}):
        raise ValueError("Study must be one directory name")
    identity = {"node": node_id, "pid": os.getpid(), "role": "cpu_embeddings",
                "start_ticks": int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])}
    status_path = node_root / "runtime/two_node_20261024/cpu_embeddings.json"
    seen: set[Path] = set()
    results: dict[str, dict] = {}
    state = {**identity, "status": "polling", "studies": results}
    with _claim(node_root / "locks/cpu_embeddings.worker.lock") as owned:
        if not owned:
            raise RuntimeError(f"CPU embedding worker already owns node {node_id}")
        try:
            while not stopped.is_set():
                state.pop("study", None)
                state.update(status="polling", consumed=len(seen))
                write_json(status_path, state)
                directories = [queue_dir / study] if study is not None else sorted(queue_dir.iterdir())
                for directory in directories:
                    if stopped.is_set():
                        break
                    try:
                        if not directory.is_dir():
                            continue
                        packets = sorted(path for path in directory.iterdir()
                                         if path.is_dir() and (path / "ready.json").is_file())
                        packets = [packet for packet in packets if packet not in seen]
                        if not packets:
                            continue
                        state.update(status="processing", study=directory.name)
                        write_json(status_path, state)
                        for packet in packets:
                            ready = json.loads((packet / "ready.json").read_text())
                            if not isinstance(ready, dict):
                                raise ValueError("Delivery ready must be a JSON object")
                            if not isinstance(ready.get("study"), str) or ready["study"] != directory.name:
                                raise ValueError("Queue study directory differs from delivery study")
                        result = process_study_embeddings(packets, node_id=node_id, node_root=node_root)
                        if result["study"] != directory.name:
                            raise ValueError("Queue study directory differs from delivery study")
                        results[directory.name] = result
                        if result["embeddings"] in {"complete", "present", "superseded"}:
                            seen.update(packets)
                    except RECOVERABLE as exc:
                        results[directory.name] = {"error": f"{type(exc).__name__}: {exc}"}
                    write_json(status_path, state)
                if once:
                    break
                stopped.wait(poll_seconds)
        finally:
            state.update(status="stopped", consumed=len(seen))
            state.pop("study", None)
            write_json(status_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=NODE_CONFIG, required=True)
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--node-root", type=Path, required=True,
                        help="Explicit project root for the logical owner")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--study")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    root = args.node_root
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run_embeddings(node_id=args.node, queue_dir=args.queue_dir or root / "transfer/study_deliveries",
                   node_root=root, once=args.once, poll_seconds=args.poll_seconds,
                   stopped=stopped, study=args.study)
    return 0


if __name__ == "__main__":
    sys.exit(main())
