"""Build and optionally execute a separate INT_Donor_E segmentation candidate.

Input: current image.tiff/channels.parquet and the existing local cpsam_v2 weights.
Output: a candidate signal, task, mask and result under the user-decision evidence.
The candidate does not replace the original mask or restore experiment membership.
"""

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import tifffile

from cellweft.data.contracts import PreparationSettings
from cellweft.data.io.readers import open_source, tiff_source
from cellweft.data.preparation.segmentation import (
    CellposeExecutor, build_segmentation_input, content_identity,
)


ROOT = Path('/mnt/NAS_21T/ProjectData/CellWeft')
SOURCE = ROOT / 'processed/Allam2024_IMC_NormalSLO/images/INT_Donor_E'
OUTPUT = ROOT / ('processed/_catalog/public/evidence/segmentation_followup_20261119/'
                 'user_decisions_20261119/INT_Donor_E_candidate')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', action='store_true')
args = parser.parse_args()
OUTPUT.mkdir(parents=True, exist_ok=True)

if args.run:
    import torch
    torch.set_num_threads(4)
    task = json.loads((OUTPUT / 'task.json').read_text())
    with CellposeExecutor('cuda:0', precision='bf16') as executor:
        print(executor.execute(task), flush=True)
else:
    metadata = json.loads((SOURCE / 'metadata.json').read_text())
    task = copy.deepcopy(metadata['segmentation_task'])
    channels = pd.read_parquet(SOURCE / 'channels.parquet')
    settings = PreparationSettings(cache_dir=OUTPUT)
    with tifffile.TiffFile(SOURCE / 'image.tiff') as tif:
        page_count = len(tif.pages)
    with open_source(tiff_source(SOURCE / 'image.tiff', page_count)) as reader:
        signal = build_segmentation_input(reader, channels, OUTPUT / 'signal.tiff', settings)
    if signal['gap']:
        raise ValueError(signal['gap'])
    weights = ROOT / 'raw/shared/pretrained/cellpose/cpsam_v2/checkpoint'
    task.update(task_id='INT_Donor_E_user_review_20261119_candidate_01',
                input_path=str(OUTPUT / 'signal.tiff'), result_path=str(OUTPUT / 'result.json'),
                weights=str(weights), signal=signal)
    # Provisional lower detection cutoff explores the sparse original mask.
    task['parameters'].update(cellprob_threshold=-1.0, min_size=10, batch_size=4)
    task['content_identity'] = {'input': content_identity(OUTPUT / 'signal.tiff'),
                                'weights': content_identity(weights)}
    task['request']['segmentation'].update(parameters=task['parameters'],
        weights={'path': str(weights), 'size': weights.stat().st_size,
                 'mtime_ns': weights.stat().st_mtime_ns})
    task['candidate_status'] = 'provisional; requires visual review and matching downstream products'
    (OUTPUT / 'task.json').write_text(json.dumps(task, ensure_ascii=False, indent=2) + '\n')
    print(OUTPUT / 'task.json', flush=True)
