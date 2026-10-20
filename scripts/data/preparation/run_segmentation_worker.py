"""Consume explicit Cellpose-SAM tasks with one persistent model per worker.

Inputs: task JSON files and corresponding two-page signal TIFFs, local cpsam_v2 weights.
Outputs: mask.tiff and result.json per task; copy the pair back to the requested result directory.
Prerequisite: dataset preparation has produced the tasks and normalized signals.
"""
import argparse
import json
from pathlib import Path

from cellweft.data.preparation.segmentation import CellposeExecutor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("tasks", type=Path, nargs="+")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--weights", type=Path, help="Local transport override for the same checkpoint")
parser.add_argument("--input", type=Path, help="Signal TIFF override for a single transported task")
parser.add_argument("--result", type=Path, help="Result JSON override for a single transported task")
parser.add_argument("--core-size", type=int, default=1024)
parser.add_argument("--halo", type=int, default=128)
parser.add_argument("--match-iou", type=float, default=0.5)
args = parser.parse_args()
if len(args.tasks) != 1 and (args.input or args.result):
    parser.error("Transport path overrides require one task")

# Decode signal tiles, run native-scale inference, and return globally labeled masks.
with CellposeExecutor(args.device,args.core_size,args.halo,args.match_iou) as executor:
    for path in args.tasks:
        task = json.loads(path.read_text())
        if args.weights:
            task["weights"] = str(args.weights)
        print(executor.execute(task,input_path=args.input,result_path=args.result))
