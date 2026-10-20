"""Synthetic label-contract checks only: no source pixels, preprocessing or model calls."""
import numpy as np
from pilot_io import instance_metrics, cell_statistics

x = np.array([[0, 11, 11, 0, 0], [0, 11, 11, 0, 25], [0, 0, 0, 0, 25]], dtype=np.uint32)
y = np.where(x == 11, 90000000, np.where(x == 25, 3, 0)).astype(np.uint32)
assert instance_metrics(x, y)['f1'] == 1
assert instance_metrics(x, y)['foreground_iou'] == 1
assert cell_statistics(x, 0.5)['count'] == 2
assert cell_statistics(x, 0.5)['border_fraction'] == 1.0
zero = np.zeros((2, 4), np.uint32)
assert instance_metrics(zero, zero)['f1'] is None
assert instance_metrics(zero, zero)['foreground_iou'] == 1
assert instance_metrics(zero, np.ones_like(zero))['precision'] == 0
assert instance_metrics(np.ones_like(zero), zero)['recall'] == 0
# Exact IoU=.5 split: two candidate matches compete for one true instance.
truth = np.ones((2, 4), np.uint32)
split = np.tile(np.array([7, 7, 9, 9], np.uint32), (2, 1))
r = instance_metrics(truth, split)
assert (r['matched_iou_0_5'], r['precision'], r['recall']) == (1, 0.5, 1)
# No background pixels and disjoint foreground preserve correct instance counting.
assert instance_metrics(truth, truth * 999)['f1'] == 1
left = np.array([[1, 0], [0, 0]], np.uint32)
right = np.array([[0, 0], [0, 8]], np.uint32)
assert instance_metrics(left, right)['f1'] == 0
assert instance_metrics(left, right)['foreground_iou'] == 0
# One-to-one cardinality at the threshold for a merge is symmetric in F1.
assert instance_metrics(split, truth)['matched_iou_0_5'] == 1
assert instance_metrics(split, truth)['f1'] == r['f1']
print('PASS: permutation, sparse IDs, empty/no-background, exact threshold, split/merge and disjoint cases')

# Strictly below the inclusive threshold must fail the edge test.
assert instance_metrics(np.ones((1, 5), np.uint32), np.array([[8, 8, 0, 0, 0]], np.uint32))['matched_iou_0_5'] == 0
assert cell_statistics(x, None)['area_um2_median'] is None
assert cell_statistics(x, None)['area_px_median'] == 3
from pilot_io import validate_fit_sources, load_samples
import copy
import json
import tempfile
from pathlib import Path

source = {'kind': 'ims', 'path': '/synthetic/ibex.ims', 'indices': [0, 4], 'z_index': 2}
sample = {'sample_id': 'ibex', 'study': 'Radtke_IBEX', 'technology': 'mIF', 'mpp': None,
          'source': source, 'roi': [0, 0, 10, 10], 'fit_scope': 'same_source_disjoint_windows',
          'fit_sources': [{**source, 'roi': roi} for roi in [[10, 0, 10, 10], [0, 10, 10, 10], [10, 10, 10, 10]]]}
assert validate_fit_sources(sample) == 'same_source_disjoint_windows'


def rejects(value: dict) -> None:
    try:
        validate_fit_sources(value)
    except ValueError:
        return
    raise AssertionError('Invalid fit definition was accepted')


bad = copy.deepcopy(sample)
bad['fit_sources'][0]['roi'] = [9, 0, 10, 10]
rejects(bad)  # Target overlap.
bad = copy.deepcopy(sample)
bad['fit_sources'][2]['roi'] = [10, 9, 10, 10]
rejects(bad)  # Fit-to-fit overlap.
for key, value in [('indices', [0, 5]), ('z_index', 3), ('path', '/synthetic/other.ims')]:
    bad = copy.deepcopy(sample)
    bad['fit_sources'][0][key] = value
    rejects(bad)
for count in [2, 4]:
    bad = copy.deepcopy(sample)
    bad['fit_sources'] = (bad['fit_sources'] * 2)[:count]
    rejects(bad)
bad = copy.deepcopy(sample)
bad['study'] = 'other'
bad['sample_id'] = 'other'
rejects(bad)
bad = copy.deepcopy(sample)
bad['fit_scope'] = 'independent_files'
rejects(bad)
independent = copy.deepcopy(sample)
independent['fit_scope'] = 'independent_files'
for i, fit in enumerate(independent['fit_sources']):
    fit['path'] = f'/synthetic/independent_{i}.ims'
assert validate_fit_sources(independent) == 'independent_files'
independent['fit_sources'][1]['path'] = independent['fit_sources'][0]['path']
rejects(independent)
with tempfile.TemporaryDirectory(prefix='cellweft_synthetic_') as tmp:
    path = Path(tmp) / 'manifest.json'
    path.write_text(json.dumps({'samples': [sample]}))
    assert load_samples(path)[0]['mpp'] is None
print('PASS: fit count, explicit IBEX scope, target/fit overlaps, selector/Z identity, independent files and null mpp')
