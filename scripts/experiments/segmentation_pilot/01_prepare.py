"""Prepare fixed A/B two-channel inputs from a frozen manifest.
Inputs: frozen manifest + 1..3 fit regions; explicit IBEX spatial-holdout exception.
Outputs: prepared/<sample_id>/{raw_selected.npz,input_A.npy,input_B.npy,stats.json,reference.npy}.
Target ROI is excluded from fitting; IBEX may use disjoint windows in its source file.
All percentiles are exact within the bounded read region, including background.
"""
import os
import resource
import time
import traceback
for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "8"
import numpy as np
from scipy.ndimage import gaussian_filter
from pilot_io import arguments, load_samples, read_source, read_reference, write_json, source_files, validate_fit_sources


def p1p99(image: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Fixed model-interface adaptation; degenerate composite becomes zero."""
    low, high = np.percentile(image, [1, 99])
    scaled = np.clip((image - low) / (high - low), 0, 1) if high > low else np.zeros_like(image)
    return scaled.astype(np.float32), [float(low), float(high)]


def diagnostics(image: np.ndarray, names: list[str], channel_axis: int = 0) -> list[dict]:
    """All pixels including background, before processing or at the final interface."""
    rows = []
    for name, plane in zip(names, np.moveaxis(image, channel_axis, 0)):
        values = np.percentile(plane, [0, 50, 90, 99, 100])
        rows.append({'channel': name, **dict(zip(['min', 'p50', 'p90', 'p99', 'max'], values.tolist())),
                     'nonzero_fraction': float(np.mean(plane != 0))})
    return rows


args = arguments(__doc__)
samples = load_samples(args.manifest)
np.random.seed(20261019)

# Fit pooled study/channel moments, one predetermined read region at a time.
def prepare_sample(sample: dict) -> None:
    sample_id = sample['sample_id']
    fit_sources = sample.get('fit_sources', [])
    fit_scope = validate_fit_sources(sample)
    nc = len(sample['channel_names'])
    if nc != len(sample['source']['indices']):
        raise ValueError(f'{sample_id}: selected channel_names/indices lengths differ')
    seen = set()
    count = np.zeros(nc, dtype=np.int64)
    mean, m2 = np.zeros(nc), np.zeros(nc)
    fit_records = []
    for source in fit_sources:
        files = source_files(source)
        seen.update(files)
        image, domain = read_source(source, source.get('roi'))
        if image.shape[0] != nc:
            raise ValueError(f'{sample_id}: fit channel count differs; controller must preserve selected marker identity/order')
        q99 = np.percentile(image, 99, axis=(1, 2))
        transformed = np.log1p(np.minimum(image, q99[:, None, None]))
        n = image.shape[1] * image.shape[2]
        for c in range(nc):
            values = transformed[c].astype(np.float64)
            local_mean = values.mean()
            local_m2 = np.square(values - local_mean).sum()
            delta = local_mean - mean[c]
            total = count[c] + n
            m2[c] += local_m2 + delta**2 * count[c] * n / total
            mean[c] += delta * n / total
            count[c] = total
        fit_records.append({'domain': domain, 'q99': q99.tolist(), 'pixels_per_channel': n})
        del image, transformed
    std = np.sqrt(m2 / count)  # Population moments of pooled log1p(P99-clipped) pixels.

    # Prepare A and B on the identical selected native-pixel domain.
    raw, domain = read_source(sample['source'], sample['roi'])
    height, width = raw.shape[1:]
    x, y, _, _ = sample['roi'] if sample['roi'] is not None else [0, 0, width, height]
    native_h, native_w = domain['native_yx']
    source_edges = {'left': x == 0, 'top': y == 0,
                    'right': x + width == native_w, 'bottom': y + height == native_h}
    mpp = sample.get('mpp')
    q1, q99 = np.percentile(raw, [1, 99], axis=(1, 2))
    reasons = {}
    for c in range(nc):
        if raw[c].max() == raw[c].min():
            reasons[c] = 'constant target channel'
        elif q99[c] <= q1[c]:
            reasons[c] = 'degenerate target P1/P99 range'
        elif std[c] == 0:
            reasons[c] = 'zero fit transformed variance'
    groups = {}
    for group in ['membrane', 'nuclear']:
        positions = sample[group + '_positions']
        if any(not isinstance(c, int) or not 0 <= c < nc for c in positions):
            raise ValueError(f'{sample_id}: invalid {group}_positions within selected channels')
        groups[group] = [c for c in positions if c not in reasons]
    if not groups['nuclear']:
        raise ValueError(f'{sample_id}: no usable nuclear channels; omissions={reasons}')

    composites = {}
    composite_ranges = {}
    for branch in ['A', 'B']:
        processed = np.zeros_like(raw)
        for c in range(nc):
            if c in reasons:
                continue
            if branch == 'A':
                standardized = (np.log1p(np.minimum(raw[c], q99[c])) - mean[c]) / std[c]
                processed[c] = gaussian_filter(standardized, sigma=1, radius=1, mode='reflect')
            else:
                processed[c] = np.clip((raw[c] - q1[c]) / (q99[c] - q1[c]), 0, 1)
        channels, ranges = [], []
        for group in ['membrane', 'nuclear']:
            composite = processed[groups[group]].mean(axis=0) if groups[group] else np.zeros(raw.shape[1:], np.float32)
            adapted, limits = p1p99(composite)
            if group == 'nuclear' and limits[1] <= limits[0]:
                raise ValueError(f'{sample_id}/{branch}: nuclear composite has degenerate P1/P99; replace sample')
            channels.append(adapted)
            ranges.append(limits)
        composites[branch] = np.stack(channels, axis=-1)
        composite_ranges[branch] = ranges

    # Save only derived artifacts; reference uses exactly the same native ROI.
    out = args.output / 'prepared' / sample_id
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'raw_selected.npz', image=raw)
    for branch, image in composites.items():
        np.save(out / f'input_{branch}.npy', image)
    reference_path = out / 'reference.npy'
    if sample['reference'] is not None:
        ref = sample['reference']
        if ref.get('alignment_verified') is not True or not ref.get('method') or not ref.get('acquisition'):
            raise ValueError('Reference requires verified acquisition/origin/axes/scale alignment and method')
        if not isinstance(ref.get('whole_cell'), bool):
            raise ValueError('Reference whole_cell must explicitly distinguish cell from nucleus labels')
        np.save(reference_path, read_reference(sample['reference'], sample['roi'], raw.shape[1:]))
    write_json(out / 'stats.json', {
        'sample': sample, 'actual_domain': domain, 'q1': q1.tolist(), 'q99': q99.tolist(),
        'physical_fov_um': [width * mpp, height * mpp] if mpp is not None else None,
        'roi_edges': {'source_native': source_edges,
                      'artificial_crop': {side: not value for side, value in source_edges.items()},
                      'model_internal_tile_edges': 'not measured by ROI border_fraction'},
        'fit_mean': mean.tolist(), 'fit_std': std.tolist(), 'fit_count': count.tolist(),
        'fit_filenames': sorted(seen), 'fit_records': fit_records,
        'fit_scope': fit_scope, 'fit_region_count': len(fit_sources),
        'statistics_domain': 'all pixels including background; pooled pixels, not equal image weights',
        'statistics_status': 'exact read-region percentiles; ROI-derived approximation to full-image P99; small-sample pooled mean/std',
        'source_intensity_phase': sample.get('source_intensity_phase', sample['source'].get('intensity_phase', 'unknown')),
        'statistics_key': {'study': sample['study'], 'panel': sample.get('panel', 'controller-defined compatible panel'),
                           'selected_markers': sample['channel_names'], 'measured_forms': sample.get('measured_forms')},
        'source_phase_interpretation': 'unknown phase permits experiment only; no claim of production-compatible one-time processing',
        'input_diagnostics': {'source_selected': diagnostics(raw, sample['channel_names']),
            **{branch: diagnostics(image, ['membrane', 'nucleus'], -1) for branch, image in composites.items()}},
        'fit_channel_identity': 'manifest controller guarantees matching marker identities and selected order',
        'omitted_channels': {sample['channel_names'][c] + f' [position {c}]': why for c, why in reasons.items()},
        'active_positions': groups, 'empty_membrane': not bool(groups['membrane']),
        'composite_p1_p99_membrane_nucleus': composite_ranges,
        'gaussian': {'sigma': 1, 'radius': 1, 'mode': 'reflect', 'axes': 'spatial YX only', 'boundary': 'ROI reflect, no physical halo'},
        'model_input': 'H,W,2 [membrane,nucleus]; final composite P1/P99 clipped [0,1]; CP normalize=False',
    })
    print(f'{sample_id}: prepared A/B {raw.shape[1:]}', flush=True)

# A run owns new prepared/runtime directories; changed settings require a new output path.
(args.output / 'prepared').mkdir(parents=True, exist_ok=False)
prep_runtime = args.output / 'runtime' / 'preparation'
prep_runtime.mkdir(parents=True, exist_ok=False)
failures = []
for sample in samples:
    start = time.perf_counter()
    record = {'sample_id': sample['sample_id'], 'status': 'failed'}
    try:
        prepare_sample(sample)
        record['status'] = 'ok'
    except Exception as exc:
        failures.append(sample['sample_id'])
        record.update(error=str(exc), traceback=traceback.format_exc())
    record['preparation_seconds'] = time.perf_counter() - start
    record['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(prep_runtime / f"{sample['sample_id']}.json", record)
write_json(prep_runtime / 'summary.json', {'manifest_n': len(samples), 'failed': failures})
if failures:
    raise SystemExit(f'Preparation failed for {failures}; retained status records, no inference attempted here')
