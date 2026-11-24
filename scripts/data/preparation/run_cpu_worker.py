"""Execute one explicit CPU task JSON and write output_dir/result.json."""
import argparse
import json
import os
from pathlib import Path


def main() -> None:
    """Set native thread budgets before loading the reusable CPU task executor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="Explicit local CPU task JSON path")
    args = parser.parse_args()
    task = json.loads(args.task.read_text())
    workers = task.get("cpu_workers", 4)
    # Invalid budgets are rejected by execute_cpu_task after withdrawing an old receipt.
    threads = str(workers if type(workers) is int and 1 <= workers <= 4 else 4)
    for name in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = threads
    import pyarrow as pa
    pa.set_cpu_count(int(threads))
    pa.set_io_thread_count(int(threads))
    from cellweft.data.preparation.cpu_tasks import execute_cpu_task
    print(json.dumps(execute_cpu_task(task), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
