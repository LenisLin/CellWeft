"""Execute a finite reviewed study batch across configured GPU and CPU nodes.

Inputs: local runtime JSON plan with explicit accepted image IDs and mask bindings.
Outputs: formal per-image NAS products, queue receipts, run status.
Side effects: owns remote queue workers and local staging; serializes formal NAS publication.
Prerequisites: deployed nodes, reviewed inputs, specified not-before UTC time.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def main() -> None:
    """Wait until the authorized start and execute one explicit finite batch."""
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='4' if key=='OMP_NUM_THREADS' else '1'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    due=datetime.fromisoformat(plan['not_before'])
    while datetime.now(timezone.utc)<due:time.sleep(max(0,min(30,(due-datetime.now(timezone.utc)).total_seconds())))
    from cellweft.data.preparation.execution import run_distributed
    from cellweft.data.io.publication import resource_lock
    with resource_lock(Path(plan['nas_lock_root']),name='distributed_controller.lock'):
        run_distributed(plan)


if __name__=='__main__':main()
