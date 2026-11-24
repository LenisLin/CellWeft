"""Consume owner-bound segmentation tasks from the shared NAS queue."""
from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

from cellweft.data.preparation.stage_queue import (
    DEVICE_COUNTS, SegmentationQueueWorker, _claim,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=DEVICE_COUNTS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, required=True,
                        help="Physical executor project root, shared by its direct and queued GPU locks")
    parser.add_argument("--weights", type=Path,
                        help="Runtime checkpoint path; content must match each immutable task")
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.device not in {f"cuda:{n}" for n in range(DEVICE_COUNTS[args.node])}:
        parser.error("Device is outside this physical node")

    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    import torch
    torch.set_num_threads(4)
    worker = SegmentationQueueWorker(physical_node=args.node, device=args.device,
                                     queue_root=args.queue_root, physical_root=args.node_root,
                                     weights_override=args.weights)
    lock = args.node_root / "locks" / f"{args.device.replace(':', '_')}.worker.lock"
    with _claim(lock) as owned:
        if not owned:
            raise RuntimeError(f"Worker already owns {args.node}/{args.device}")
        try:
            while not stopped.is_set():
                outcome = worker.run_once()
                if args.once:
                    break
                if outcome in {"idle", "busy"}:
                    stopped.wait(args.poll_seconds)
        finally:
            worker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
