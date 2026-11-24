"""Resume one study-input CLI queue and its independent review/repair chain.

Inputs: existing runtime plan.json, status.json and jobs; optional STOP_ADMISSION
or STOP files. Outputs: compatible status, job reports and execution receipts.
STOP_ADMISSION drains admitted studies; STOP waits for active tasks to close.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cellweft.data.preparation.study_input_execution import (
    ControllerConfig, MODEL, REASONING_EFFORT, StudyInputController, load_controller_status,
)
from cellweft.data.preparation.io_coordination import default_io_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10, choices=range(1, 16), metavar="1..15")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reasoning-effort", default=REASONING_EFFORT)
    parser.add_argument("--nas-lock-root", type=Path, default=default_io_root())
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--status", action="store_true", help="Read status without starting tasks")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(load_controller_status(args.run_root), ensure_ascii=False, indent=2))
        return
    lock_root = args.nas_lock_root.resolve()
    os.environ["CELLWEFT_NAS_LOCK_ROOT"] = str(lock_root)
    controller = StudyInputController(ControllerConfig(
        run_root=args.run_root, workers=args.workers, model=args.model,
        reasoning_effort=args.reasoning_effort, codex=args.codex, nas_lock_root=lock_root))
    print(json.dumps(controller.run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
