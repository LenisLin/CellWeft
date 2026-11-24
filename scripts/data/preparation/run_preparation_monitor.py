"""Read aggregate preparation status without controlling any worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time

from cellweft.data.preparation.monitoring import PreparationMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    monitor = PreparationMonitor(config)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while not stop.is_set():
        started = time.monotonic()
        monitor.snapshot()
        if args.once:
            break
        stop.wait(max(0, 30 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
