"""Poll immutable study deliveries on one node GPU, or run one smoke pass."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path

from cellweft.data.preparation.study_node import NODE_CONFIG, _claim, process_delivery
from cellweft.data.preparation.execution_profile import execution_profile
from cellweft.data.preparation.segmentation import CellposeExecutor
from cellweft.data.io.writers import write_json




def assert_direct_profile(packet: Path, node: str) -> None:
    """The persistent direct worker has one owner-default executor."""
    ready = json.loads((packet / "ready.json").read_text())
    if ready.get("node_id") != node:
        raise ValueError("Delivery belongs to another node")
    if "execution_profile" in ready:
        if ready["execution_profile"] is None:
            raise ValueError("Explicit execution profile cannot be null")
        if execution_profile(node, ready["execution_profile"]) != execution_profile(node):
            raise ValueError("Direct GPU worker cannot run a non-default execution profile")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=NODE_CONFIG, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, required=True,
                        help="Explicit project root for the logical owner")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--packet", type=Path, help="process this ready delivery only")
    parser.add_argument("--allow-unknown-mask-segmentation", action="store_true",
                        help="run cpsam_v2 when the author-mask binding is unknown")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    count, _, _ = NODE_CONFIG[args.node]
    if args.device not in {f"cuda:{i}" for i in range(count)}:
        parser.error(f"node {args.node} supports cuda:0..{count - 1}")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    root = args.node_root
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    import torch
    torch.set_num_threads(4)
    seen = set()
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    status_path = root / "runtime" / "two_node_20261024" / f"{args.device.replace(':', '_')}.json"
    identity = {"pid": os.getpid(), "node": args.node, "device": args.device,
                "start_ticks": int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])}
    with _claim(root / "locks" / f"{args.device.replace(':', '_')}.worker.lock") as owned:
        if not owned:
            raise RuntimeError(f"Worker already owns {args.node}/{args.device}")
        with CellposeExecutor(device=args.device, precision=NODE_CONFIG[args.node][1]) as segmenter:
            while not stopped.is_set():
                write_json(status_path, {**identity, "status": "polling", "consumed": len(seen)})
                packets = ([args.packet] if args.packet is not None else
                           sorted(packet for study in args.queue_dir.iterdir() if study.is_dir()
                                  for packet in study.iterdir() if packet.is_dir()))
                for packet in packets:
                    if stopped.is_set():
                        break
                    if packet in seen:
                        continue
                    if packet.is_dir() and (packet / "ready.json").is_file():
                        assert_direct_profile(packet, args.node)
                        write_json(status_path, {**identity, "status": "processing", "delivery": str(packet)})
                        summary = process_delivery(packet, node_id=args.node, node_root=root,
                                                   device=args.device, cache_root=args.cache_root,
                                                   should_stop=stopped.is_set, segmenter=segmenter,
                                                   allow_unknown_mask_segmentation=(
                                                       True if args.allow_unknown_mask_segmentation else None))
                        print(f"{summary['study']}: {summary['images']}; embeddings={summary['embeddings']}", flush=True)
                        if (set(summary["images"]) == set(summary["image_ids"])
                                and all(status in {"computed", "quality_pending", "partial", "superseded"}
                                                      for status in summary["images"].values())
                                and summary["embeddings"] in {"complete", "present", "superseded"}):
                            seen.add(packet)
                    elif args.packet is not None:
                        raise ValueError("Requested packet is not ready")
                if args.once or args.packet is not None:
                    break
                stopped.wait(args.poll_seconds)
    write_json(status_path, {**identity, "status": "stopped", "consumed": len(seen)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
