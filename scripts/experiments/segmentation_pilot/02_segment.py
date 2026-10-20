"""Run one fixed scheme serially on prepared A/B inputs from 01.
Inputs: frozen manifest, run directory, explicit local model weights.
Outputs: uint32 masks, raw CP/IS outputs, per-sample statuses and load/inference resources.
CP/IS use one torch GPU; SD_B uses CPU-only TF2.12, without torch.
"""
import importlib.metadata
import json
import os
import resource
import time
import traceback
from pathlib import Path

for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '8'

import numpy as np
from pilot_io import arguments, labels, load_samples, write_json

args = arguments(__doc__, segmentation=True)
samples = load_samples(args.manifest)
scheme = args.scheme
cpu_only = scheme == 'SD_B'
run_dir = args.output / 'runtime' / scheme
mask_dir = args.output / 'masks' / scheme
raw_dir = args.output / 'model_raw' / scheme
# Prevent mixed artifacts or silent replacement of an earlier scheme in the same run.
for folder in [run_dir, mask_dir, raw_dir]:
    folder.mkdir(parents=True, exist_ok=False)


def current_rss_bytes() -> int:
    with open('/proc/self/statm') as handle:
        return int(handle.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def resources() -> dict:
    return {'rss_bytes': current_rss_bytes(),
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'gpu': None if cpu_only else torch.cuda.get_device_name(device),
            'gpu_max_allocated_bytes': None if cpu_only else torch.cuda.max_memory_allocated(device),
            'gpu_max_reserved_bytes': None if cpu_only else torch.cuda.max_memory_reserved(device)}


# Unknown physical scales explicitly skip IS; CP and SD retain native-pixel validity.
eligible, skipped = [], []
for sample in samples:
    if scheme.startswith('IS') and sample.get('mpp') is None:
        skipped.append(sample['sample_id'])
        write_json(run_dir / f"{sample['sample_id']}.json", {
            'sample_id': sample['sample_id'], 'scheme': scheme, 'status': 'skipped',
            'reason': 'unverified mpp; IS scale is never guessed', 'gpu': None})
    else:
        eligible.append(sample)
if not eligible:
    write_json(run_dir / 'model.json', {'status': 'skipped', 'reason': 'no samples with verified IS mpp'})
    write_json(run_dir / 'summary.json', {'scheme': scheme, 'manifest_n': len(samples),
        'attempted': 0, 'ok': [], 'failed': [], 'skipped': skipped})
    raise SystemExit(0)

# Model load is separate from inference; no downloads or hidden device fallback.
try:
    weights = args.weights.resolve(strict=True)
    np.random.seed(20261019)
    if cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Must precede TensorFlow import.
        os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.random.set_seed(20261019)
        from stardist.models import StarDist2D
        if not weights.is_dir():
            raise ValueError('SD weights must be a local 2D_versatile_fluo model directory')
        load_start = time.perf_counter()
        with tf.device('/CPU:0'):
            model = StarDist2D(None, name=weights.name, basedir=str(weights.parent))
        adapter = {'package': importlib.metadata.version('stardist'), 'tensorflow': tf.__version__,
                   'device': 'CPU:0', 'torch_required': False, 'model': 'local 2D_versatile_fluo',
                   'normalization': 'prepared B nuclear input; no extra normalization',
                   'n_tiles': [4, 4], 'scale': 'native pixels', 'target': 'nuclei only'}
    else:
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261019)
        if not torch.cuda.is_available():
            raise RuntimeError('CP/IS require CUDA')
        device = torch.device('cuda:0')
        torch.cuda.synchronize(device)
        load_start = time.perf_counter()
        if scheme.startswith('CP'):
            from cellpose import models
            if importlib.metadata.version('cellpose') != '4.2.1.1':
                raise ValueError('Frozen CP adapter requires cellpose==4.2.1.1')
            if not weights.is_file():
                raise ValueError('CP requires an explicit local cpsam_v2 checkpoint')
            def no_download(*args, **kwargs):
                raise RuntimeError('Cellpose attempted automatic checkpoint download')
            models.cache_model_path = no_download
            model = models.CellposeModel(gpu=True, device=device, pretrained_model=str(weights))
            if Path(model.pretrained_model).resolve() != weights or not model.gpu:
                raise RuntimeError('Cellpose did not retain the requested checkpoint/CUDA device')
            adapter = {'package': '4.2.1.1', 'normalize': False, 'diameter': None,
                       'batch_size': 4, 'bsize': 256, 'tile_overlap': 0.1,
                       'flow_threshold': 0.4, 'cellprob_threshold': 0.0,
                       'min_size': 15, 'max_size_fraction': 0.4, 'augment': False,
                       'resample': True, 'niter': None, 'do_3D': False,
                       'scale': 'native pixels, rescale=1 with diameter=None',
                       'channels': '[membrane,nucleus,zero], HWC float32, channel_axis=-1'}
        else:
            from instanseg import InstanSeg
            module = torch.jit.load(str(weights), map_location='cpu').eval()
            if not module.cells_and_nuclei or not np.isclose(float(module.pixel_size), 0.5):
                raise ValueError('IS requires fluorescence_nuclei_and_cells with model pixel_size=0.5')
            model = InstanSeg(module, device='cuda:0', verbosity=1, channels_last=False, image_reader='skimage.io')
            adapter = {'package': importlib.metadata.version('instanseg-torch'), 'normalise': False,
                       'batch_size': 4, 'tile_size': 512, 'rescale_output': True,
                       'model_pixel_size': 0.5, 'input_mpp': 'verified per sample; unknown -> skipped', 'channels_last_memory_format': False,
                       'channels': 'two prepared channels [membrane,nucleus], CHW float32',
                       'effective_chain': 'prepared -> bilinear resample to 0.5 um/px -> IS -> nearest native-shape labels'}
        torch.cuda.synchronize(device)
    write_json(run_dir / 'model.json', {'scheme': scheme, 'weights': str(weights),
        'status': 'loaded', 'adapter_validation': 'requires actual inference; source inspection alone is not a test',
        'load_seconds': time.perf_counter() - load_start, 'adapter': adapter, **resources()})
except Exception as exc:
    error = {'scheme': scheme, 'weights': str(args.weights), 'status': 'failed',
             'stage': 'model_load', 'error': str(exc), 'traceback': traceback.format_exc()}
    write_json(run_dir / 'model.json', error)
    for sample in eligible:
        write_json(run_dir / f"{sample['sample_id']}.json", {**error, 'sample_id': sample['sample_id']})
    write_json(run_dir / 'summary.json', {'scheme': scheme, 'manifest_n': len(samples),
        'attempted': 0, 'ok': [], 'failed': [s['sample_id'] for s in eligible], 'skipped': skipped})
    raise

# Read one ROI at a time. Inference includes transfers, excludes input I/O and output writes.
failures, successful = [], []
for sample in eligible:
    sample_id = sample['sample_id']
    record = {'sample_id': sample_id, 'scheme': scheme, 'status': 'failed'}
    start = None
    try:
        prep_record = args.output / 'runtime' / 'preparation' / f'{sample_id}.json'
        if json.loads(prep_record.read_text())['status'] != 'ok':
            raise ValueError('Preparation failed; partial prepared files are not valid inputs')
        read_start = time.perf_counter()
        image = np.load(args.output / 'prepared' / sample_id / f'input_{scheme[-1]}.npy', allow_pickle=False)
        record['input_read_seconds'] = time.perf_counter() - read_start
        if image.ndim != 3 or image.shape[-1] != 2 or not np.isfinite(image).all():
            raise ValueError('Expected finite prepared H,W,2 [membrane,nucleus]')
        record['rss_before_bytes'] = current_rss_bytes()
        if cpu_only:
            start = time.perf_counter()
            with tf.device('/CPU:0'):
                mask, details = model.predict_instances(image[..., 1], n_tiles=(4, 4))
        else:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with torch.inference_mode():
                if scheme.startswith('CP'):
                    cp_input = np.concatenate([image, np.zeros_like(image[..., :1])], axis=-1)
                    mask, flows, styles = model.eval(cp_input, batch_size=4, bsize=256,
                        channel_axis=-1, normalize=False, diameter=None, tile_overlap=0.1,
                        flow_threshold=0.4, cellprob_threshold=0.0, min_size=15,
                        max_size_fraction=0.4, augment=False, resample=True, niter=None, do_3D=False)
                else:
                    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())
                    result = model.eval_medium_image(tensor, pixel_size=sample['mpp'],
                        normalise=False, batch_size=4, tile_size=512, target='all_outputs',
                        return_image_tensor=False, rescale_output=True).detach().cpu().numpy()
                    if result.ndim != 4 or result.shape[:2] != (1, 2):
                        raise ValueError(f'IS expected [1,2,H,W] nuclei,cells; got {result.shape}')
                    if (not np.isfinite(result).all() or np.any(result != np.floor(result))
                            or np.any(result < 0) or np.any(result > np.iinfo(np.uint32).max)):
                        raise ValueError('IS labels must be finite nonnegative uint32-valued integers')
                    nuclei = labels(result[0, 0].astype(np.uint32), image.shape[:2])
                    mask = result[0, 1].astype(np.uint32)
            torch.cuda.synchronize(device)
        record['inference_seconds'] = time.perf_counter() - start
        mask = labels(mask, image.shape[:2])
        np.save(mask_dir / f'{sample_id}.npy', mask)
        if scheme.startswith('CP'):
            np.savez_compressed(raw_dir / f'{sample_id}.npz', mask=mask,
                flow_rgb=flows[0], flow_xy=flows[1], cell_probability=flows[2], styles=styles)
        elif scheme.startswith('IS'):
            np.save(mask_dir / f'{sample_id}_nuclei.npy', nuclei)
            np.savez_compressed(raw_dir / f'{sample_id}.npz', labels=result)
        record['status'] = 'ok'
        successful.append(sample_id)
    except Exception as exc:
        failures.append(sample_id)
        record.update(error=str(exc), traceback=traceback.format_exc())
        if start is not None:
            record['elapsed_until_failure_seconds'] = time.perf_counter() - start
    record.update(resources())
    write_json(run_dir / f'{sample_id}.json', record)
    print(f'{scheme}/{sample_id}: {record["status"]}', flush=True)
write_json(run_dir / 'summary.json', {'scheme': scheme, 'manifest_n': len(samples),
    'attempted': len(eligible), 'ok': successful, 'failed': failures, 'skipped': skipped})
if failures:
    raise SystemExit(f'{scheme}: {len(failures)} failures; see {run_dir}')
