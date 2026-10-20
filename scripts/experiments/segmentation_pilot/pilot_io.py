"""Shared ROI readers, manifest loading, label checks and sparse instance metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

MAX_SIDE = 1536


def arguments(description: str, segmentation: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    if segmentation:
        parser.add_argument('--scheme', choices=['CP_A', 'CP_B', 'IS_A', 'IS_B', 'SD_B'], required=True)
        parser.add_argument('--weights', type=Path, required=True)
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples = json.loads(path.read_text())['samples']
    ids = [s['sample_id'] for s in samples]
    if len(set(ids)) != len(ids) or any(Path(s).name != s or s in {'.', '..', ''} for s in ids):
        raise ValueError('sample_id must be unique, nonempty path-safe basenames')
    for sample in samples:
        mpp = sample.get('mpp')
        if mpp is not None and (not np.isfinite(mpp) or mpp <= 0):
            raise ValueError(f"{sample['sample_id']}: verified mpp must be positive; use null when unknown")
    return samples


def source_files(source: dict[str, Any]) -> set[str]:
    return {str(Path(p).resolve()) for p in source.get('paths', [source['path']])}


def validate_fit_sources(sample: dict[str, Any]) -> str:
    """Enforce independent files or the explicit three-window IBEX exception.

    Same-source windows are experimental spatial holdouts, never independent images.
    Acquisition/panel/marker/processing compatibility is frozen by the controller.
    """
    fits = sample.get('fit_sources', [])
    if not 1 <= len(fits) <= 3:
        raise ValueError('A requires 1..3 fit sources/windows')
    scope = sample.get('fit_scope', 'independent_files')
    target = sample['source']
    if scope == 'same_source_disjoint_windows':
        if 'ibex' not in (sample['study'] + ' ' + sample['technology'] + ' ' + sample['sample_id'] + ' ' + sample.get('assay', '')).lower():
            raise ValueError('same_source_disjoint_windows is restricted to explicitly identified IBEX')
        if len(fits) != 3:
            raise ValueError('IBEX exception requires exactly three fixed fit windows')
        selector = lambda s: (s['kind'], tuple(sorted(source_files(s))), tuple(s['indices']),
                              tuple(s.get('paths', [])), s.get('z_index'), s.get('maxproj', False))
        if any(selector(s) != selector(target) for s in fits):
            raise ValueError('IBEX fit windows must use identical source files, indices and Z selection')
        windows = [sample.get('roi')] + [s.get('roi') for s in fits]
        for window in windows:
            if window is None or len(window) != 4:
                raise ValueError('IBEX target and fit windows require explicit xywh')
            x, y, w, h = window
            region((y + h, x + w), window)
        for i, (x, y, w, h) in enumerate(windows):
            for xx, yy, ww, hh in windows[i + 1:]:
                if x < xx + ww and xx < x + w and y < yy + hh and yy < y + h:
                    raise ValueError('IBEX fit/target windows overlap')
    elif scope == 'independent_files':
        seen = source_files(target)
        for source in fits:
            files = source_files(source)
            if seen & files:
                raise ValueError('independent_files excludes target and repeated fit files')
            seen.update(files)
    else:
        raise ValueError(f'Unknown fit_scope: {scope}')
    return scope


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def region(shape: tuple[int, ...], roi: list[int] | None) -> tuple[slice, slice]:
    """Validate fixed native-pixel [x,y,w,h]; reject oversized full-image reads."""
    h, w = shape[-2:]
    x, y, rw, rh = [0, 0, w, h] if roi is None else roi
    if any(int(v) != v for v in (x, y, rw, rh)):
        raise ValueError('ROI coordinates must be integers')
    if min(x, y) < 0 or min(rw, rh) <= 0 or x + rw > w or y + rh > h:
        raise ValueError(f'ROI {roi} outside native YX={h,w}')
    if max(rw, rh) > MAX_SIDE:
        raise ValueError(f'Read region {rw}x{rh} exceeds 1536; freeze a smaller ROI in manifest')
    return slice(y, y + rh), slice(x, x + rw)


def _z_selection(source: dict[str, Any], depth: int) -> int | range:
    if 'z_index' in source and source.get('maxproj', False):
        raise ValueError('Choose z_index OR explicit maxproj, not both')
    if 'z_index' in source:
        z = source['z_index']
        if not isinstance(z, int) or not 0 <= z < depth:
            raise ValueError(f'z_index {z} outside depth {depth}')
        return z
    if source.get('maxproj') is True:
        return range(depth)
    if depth == 1:
        return 0
    raise ValueError('Multiplane source requires explicit z_index or maxproj:true')


def _tiff_plane(path: str, index: int, roi: list[int] | None) -> tuple[np.ndarray, tuple[int, int]]:
    """Read one TIFF IFD; large planes use tile-aware aszarr slicing, never full decode."""
    import tifffile
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[index]
        if len(page.shape) != 2:
            raise ValueError(f'{path} IFD {index}: expected a scalar YX plane, got {page.shape}')
        ys, xs = region(page.shape, roi)
        if max(page.shape) > MAX_SIDE:
            import zarr
            store = page.aszarr()
            try:
                arr = zarr.open(store, mode='r')
                image = np.asarray(arr[ys, xs])
            finally:
                store.close()
        else:
            image = page.asarray()[ys, xs]
        return image, tuple(page.shape)


def _validate_intensity(image: np.ndarray, path: str) -> None:
    if not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError(f'{path}: nonfinite/negative intensity; inspect source processing, no imputation applied')


def read_source(source: dict[str, Any], roi: list[int] | None) -> tuple[np.ndarray, dict[str, Any]]:
    """Return selected C,Y,X and read domain. Native indices are never inferred.

    Zarr arrays must be CYX or CZYX at the explicit /0 array path. TIFF stack
    indices are native IFD numbers; for z-stacks the controller supplies selected
    planes as tiff_planes. IMS Z projection streams one ROI plane at a time.
    """
    kind = source['kind']
    indices = source['indices']
    if not indices or any(not isinstance(i, int) or i < 0 for i in indices):
        raise ValueError('source.indices must contain selected nonnegative native indices')
    planes = []
    if kind == 'zarr':
        import zarr
        arr = zarr.open_array(source['path'], mode='r', zarr_format=3)
        if arr.ndim not in (3, 4):
            raise ValueError(f'Expected explicit CYX/CZYX zarr array, got {arr.shape}')
        ys, xs = region(arr.shape, roi)
        native_yx = tuple(arr.shape[-2:])
        if arr.ndim == 3:
            image = np.asarray(arr.oindex[indices, ys, xs])
        else:
            z = _z_selection(source, arr.shape[1])
            if isinstance(z, int):
                image = np.asarray(arr.oindex[indices, z, ys, xs])
            else:
                image = None
                for zi in z:
                    plane = np.asarray(arr.oindex[indices, zi, ys, xs])
                    _validate_intensity(plane, source['path'])
                    image = plane if image is None else np.maximum(image, plane)
    elif kind in {'tiff_stack', 'tiff_planes'}:
        paths = source.get('paths')
        if kind == 'tiff_planes' and (paths is None or len(paths) != len(indices)):
            raise ValueError('tiff_planes.paths must list selected files in selected-channel order')
        native_yx = None
        for position, index in enumerate(indices):
            path = paths[position] if kind == 'tiff_planes' else source['path']
            plane, shape = _tiff_plane(path, 0 if kind == 'tiff_planes' else index, roi)
            if native_yx is not None and shape != native_yx:
                raise ValueError('Selected TIFF planes have unequal native YX shapes')
            native_yx = shape
            planes.append(plane)
        image = np.stack(planes)
    elif kind == 'ims':
        import h5py
        with h5py.File(source['path'], 'r') as handle:
            native_yx = None
            for index in indices:
                dataset = handle[f'DataSet/ResolutionLevel 0/TimePoint 0/Channel {index}/Data']
                logical_shape = source.get('valid_shape', dataset.shape)
                ys, xs = region(tuple(logical_shape), roi)
                shape = tuple(logical_shape[-2:])
                if native_yx is not None and shape != native_yx:
                    raise ValueError('IMS selected channels have unequal native shapes')
                native_yx = shape
                z = _z_selection(source, dataset.shape[0])
                if isinstance(z, int):
                    plane = np.asarray(dataset[z, ys, xs])
                else:
                    plane = None
                    for zi in z:
                        current = np.asarray(dataset[zi, ys, xs])
                        _validate_intensity(current, source['path'])
                        plane = current if plane is None else np.maximum(plane, current)
                planes.append(plane)
        image = np.stack(planes)
    else:
        raise ValueError(f'Unsupported source kind: {kind}')
    image = np.asarray(image, dtype=np.float32)
    _validate_intensity(image, source['path'])
    return image, {'native_yx': list(native_yx), 'roi_xywh': roi,
                   'read_yx': list(image.shape[-2:]), 'source': source,
                   'percentile_domain': 'exact within read region, independently per image/channel'}


def labels(mask: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    if mask.ndim != 2 or mask.dtype.kind not in 'iu' or np.any(mask < 0):
        raise ValueError('Mask must be nonnegative 2D integer instances, background=0')
    if shape is not None and mask.shape != shape:
        raise ValueError(f'Mask YX {mask.shape} != input YX {shape}; alignment must be resolved upstream')
    if mask.max(initial=0) > np.iinfo(np.uint32).max:
        raise ValueError('Instance ID exceeds uint32')
    return mask.astype(np.uint32, copy=False)


def read_reference(ref: dict[str, Any], roi: list[int] | None, shape: tuple[int, int]) -> np.ndarray:
    """Reference must share native coordinates; no automatic registration or resizing."""
    if ref['kind'] == 'tiff':
        mask, _ = _tiff_plane(ref['path'], 0, roi)
    elif ref['kind'] == 'npz':
        # NPZ cannot provide bounded ROI decompression: controller must supply small masks.
        import zipfile
        key = ref.get('key', 'mask')
        with zipfile.ZipFile(ref['path']) as archive:
            if archive.getinfo(key + '.npy').file_size > MAX_SIDE**2 * 8 + 4096:
                raise ValueError('Large compressed NPZ reference: export a bounded TIFF ROI upstream')
        with np.load(ref['path'], allow_pickle=False) as data:
            mask = data[key]
        if mask.ndim != 2:
            raise ValueError('Reference NPZ must contain a 2D label array')
        ys, xs = region(mask.shape, roi)
        mask = mask[ys, xs]
    else:
        raise ValueError('reference.kind must be tiff or npz')
    return labels(mask, shape)


def instance_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Maximum-cardinality one-to-one IoU>=0.5 matching using sparse overlap edges.

    IDs are compacted with unique rather than sized by max(label); no dense N×M
    allocation. Precision denominator is prediction count; empty denominators
    are null, both-empty foreground IoU is 1. Pair calls measure agreement.
    """
    truth = labels(truth)
    prediction = labels(prediction, truth.shape)
    tids, ti, ta = np.unique(truth, return_inverse=True, return_counts=True)
    pids, pi, pa = np.unique(prediction, return_inverse=True, return_counts=True)
    ti, pi = ti.ravel(), pi.ravel()
    nt, npred = int(np.sum(tids > 0)), int(np.sum(pids > 0))
    both = (truth.ravel() > 0) & (prediction.ravel() > 0)
    pairs, intersection = np.unique(np.stack([ti[both], pi[both]], axis=1), axis=0, return_counts=True)
    if pairs.size:
        iou = intersection / (ta[pairs[:, 0]] + pa[pairs[:, 1]] - intersection)
        pairs = pairs[iou >= 0.5]
    # Background may be absent; searchsorted maps only positive IDs to graph indices.
    rows = np.searchsorted(tids[tids > 0], tids[pairs[:, 0]]) if pairs.size else []
    cols = np.searchsorted(pids[pids > 0], pids[pairs[:, 1]]) if pairs.size else []
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nt, npred)).tocsr()
    graph.sort_indices()  # Ascending positive IDs fix traversal order for tied edges.
    matched = int(np.sum(maximum_bipartite_matching(graph, perm_type='column') >= 0)) if nt and npred else 0
    union = int(np.count_nonzero((truth > 0) | (prediction > 0)))
    return {'n_reference': nt, 'n_prediction': npred, 'matched_iou_0_5': matched,
            'precision': matched / npred if npred else None,
            'recall': matched / nt if nt else None,
            'f1': 2 * matched / (nt + npred) if nt + npred else None,
            'foreground_iou': int(both.sum()) / union if union else 1.0}


def cell_statistics(mask: np.ndarray, mpp: float | None) -> dict[str, Any]:
    mask = labels(mask)
    ids, counts = np.unique(mask, return_counts=True)
    area = counts[ids > 0]
    border = np.unique(np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]))
    result = {'count': int(len(area)), 'coverage': float(np.mean(mask > 0)),
              'border_fraction': float(np.sum(border > 0) / len(area)) if len(area) else None}
    for name, value in zip(['min', 'median', 'p90'], np.percentile(area, [0, 50, 90]) if len(area) else [None]*3):
        result[f'area_px_{name}'] = float(value) if value is not None else None
        result[f'area_um2_{name}'] = float(value * mpp**2) if value is not None and mpp is not None else None
    return result
