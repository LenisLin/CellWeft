"""Inspect every indexed image with a canvas axis below 224 pixels.

Inputs: current study indices, per-ROI image/mask/channel/cell assets and metadata.
Outputs: current QC registry and PNG figures under processed/_catalog/public/.
This diagnostic does not approve sampling, alter scientific assets or edit HTML.
"""

import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tifffile
from PIL import Image, ImageDraw, ImageFont

from cellweft.data.io.readers import open_source, tiff_source
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.crops import grid_origins
from cellweft.data.preparation.sampling import estimate_tissue_region


ROOT = Path('/mnt/NAS_21T/ProjectData/CellWeft')
PROCESSED = ROOT / 'processed'
CATALOG = PROCESSED / '_catalog/public'
REGISTRY = CATALOG / 'quality_control.json'
assert REGISTRY.is_file(), 'Consolidate the existing QC records with script 16 first.'
OUTPUT = CATALOG / 'quality_control/small_fov'
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / 'figures').mkdir(exist_ok=True)
manifest = json.loads((PROCESSED / '_catalog/public/experiments.json').read_text())
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
small_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
records, thumbnails = [], []
index_count = total_count = 0


def gray(values):
    """Display normalized signal without changing native coordinates or extent."""
    return np.repeat((np.clip(values, 0, 1)*255).astype(np.uint8)[..., None], 3, axis=2)


def maximum(values, pages, shape):
    """An absent signal group remains a blank display, with its markers recorded."""
    return values[pages].max(axis=0) if pages else np.zeros(shape, np.float32)


for index in sorted(PROCESSED.glob('*/images.parquet')):
    study = index.parent.name
    images = pq.read_table(index).to_pylist()
    index_count += 1
    total_count += len(images)
    dataset = manifest['datasets'][study]
    splits = {roi: split for g in dataset.get('groups', {}).values()
              for split, ids in g['members'].items() for roi in ids}
    splits.update({roi: split for split, ids in dataset.get('ungrouped', {}).items() for roi in ids})
    for image in images:
        h, w = image['height_px'], image['width_px']
        if min(h, w) >= 224:
            continue
        roi = image['image_id']
        directory = index.parent / 'images' / roi
        metadata = json.loads((directory / 'metadata.json').read_text())
        source_record = metadata['request']['image']['source']
        channels = pd.read_parquet(directory / 'channels.parquet')
        excluded = dataset.get('excluded_markers_by_image', {}).get(roi, [])
        effective = channels[channels.tiff_page.notna() & ~channels.marker_id.isin(excluded)]
        selected = {kind: effective.loc[effective[kind], 'tiff_page'].astype(int).tolist()
                    for kind in ['nuclear', 'body']}
        selected['protein'] = effective.loc[effective.role.eq('downstream'), 'tiff_page'].astype(int).tolist()
        with tifffile.TiffFile(directory / 'image.tiff') as tif:
            assert all(page.shape == (h, w) for page in tif.pages)
            values = np.stack([page.asarray() for page in tif.pages])
        with tifffile.TiffFile(directory / 'mask.tiff') as tif:
            mask = tif.pages[0].asarray()
            assert mask.shape == (h, w)
        bounds = source_record['valid_bounds']
        source = replace(tiff_source(directory / 'image.tiff', len(values)),
                         valid_bounds=tuple(bounds) if bounds is not None else None,
                         valid_mask=Path(source_record['valid_mask']) if source_record['valid_mask'] else None,
                         sample_region=Path(source_record['sample_region']) if source_record.get('sample_region') else None)
        diagnostic_tissue = None
        with open_source(source) as reader:
            valid = reader.valid_window((0, 0, w, h))
            windows = grid_origins((h, w))
            assert windows == [(0, 0)]
            measured_area = int(valid.sum())
            upper_bound = measured_area / (224*224)
            protein_nonzero = any(np.any(values[page][valid] != 0) for page in selected['protein'])
            effective_pages = sorted(set(effective.tiff_page.astype(int)))
            nonfinite = int((~np.isfinite(values[effective_pages][:, valid])).sum())
            tissue_metadata, tissue_gap, tissue_area = None, None, None
            if upper_bound <= .50:
                decision = 'insufficient_measured_area'
            elif not protein_nonzero:
                decision = 'no_nonzero_downstream_signal'
            else:
                with TemporaryDirectory(prefix='cellweft_tissue_') as scratch, \
                        estimate_tissue_region(reader, effective, Path(scratch)) as tissue:
                    tissue_metadata, tissue_gap = tissue.metadata, tissue.gap
                    if tissue.path is None:
                        decision = 'tissue_not_estimable'
                    else:
                        with tifffile.TiffFile(tissue.path) as tif:
                            diagnostic_tissue = (tif.pages[0].asarray() > 0) & valid
                        tissue_area = int(diagnostic_tissue.sum())
                        decision = ('candidate_tissue_pass_unreviewed' if tissue_area > 25088
                                    else 'candidate_tissue_fail_unreviewed')
        cells = pq.read_table(directory / 'cells.parquet', columns=['cell_index', 'mask_label', 'crop_origins']).to_pylist()
        labels = set(int(x) for x in np.unique(mask) if x > 0)
        assert labels == {r['mask_label'] for r in cells}
        assert all(r['crop_origins'] == [{'x0': 0, 'y0': 0}] for r in cells)
        assert not nonfinite
        assert not np.any(mask[~valid])
        signal_groups = {kind: maximum(values, pages, (h, w)) for kind, pages in selected.items()}
        colors = np.zeros((h, w, 3), np.uint8)
        positive = mask > 0
        hashed = mask.astype(np.uint64)*2654435761+2246822519
        for c, shift in enumerate([8, 17, 25]):
            colors[..., c][positive] = (72+((hashed[positive] >> shift) % 184)).astype(np.uint8)
        composite = np.zeros((h, w, 3), np.uint8)
        composite[..., 0] = (np.clip(signal_groups['body'], 0, 1)*255).astype(np.uint8)
        composite[..., 2] = (np.clip(signal_groups['nuclear'], 0, 1)*255).astype(np.uint8)
        edges = np.zeros((h, w), bool)
        edges[1:] |= mask[1:] != mask[:-1]
        edges[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        composite[edges] = 255
        panels = [('Nuclear max', gray(signal_groups['nuclear'])),
                  ('Protein max', gray(signal_groups['protein'])),
                  ('Signals + original contours', composite),
                  ('Original instance mask', colors),
                  ('Measured domain', gray(valid)),
                  ('Candidate tissue (not approved)' if diagnostic_tissue is not None
                   else 'Tissue not evaluated: area cap',
                   gray(diagnostic_tissue) if diagnostic_tissue is not None else np.zeros((h,w,3),np.uint8))]
        canvas = Image.new('RGB', (752, 624), '#12181d')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 6), study, font=font, fill='white')
        draw.text((12, 27), roi, font=font, fill='white')
        draw.text((12, 48), f'Canvas {w}x{h}; measured cap {upper_bound:.2%}; {decision}', font=small_font, fill='#bdccd5')
        for i, (title, pixels) in enumerate(panels):
            x, y = 12+(i%3)*248, 78+(i//3)*268
            draw.text((x, y), title, font=small_font, fill='white')
            canvas.paste(Image.fromarray(pixels), (x, y+24))
        figure = OUTPUT / 'figures' / f'{study}__{roi}.png'
        canvas.save(figure)
        tile = Image.new('RGB', (240, 285), '#12181d')
        tdraw = ImageDraw.Draw(tile)
        tdraw.text((7, 0), study.split('_')[0], font=small_font, fill='white')
        tdraw.text((7, 17), roi[:31], font=small_font, fill='white')
        tile.paste(Image.fromarray(gray(signal_groups['protein'])), (7, 39))
        tdraw.text((7, 263), f'{w}x{h}; cap {upper_bound:.1%}', font=small_font, fill='white')
        thumbnails.append(tile)
        records.append({'study': study, 'roi': roi, 'split': splits.get(roi, 'excluded'),
            'canvas_shape_yx': [h, w], 'source_shape_yx': [image['source_y1_px']-image['source_y0_px'], image['source_x1_px']-image['source_x0_px']],
            'padding_ltrb': [image[k] for k in ['pad_left_px','pad_top_px','pad_right_px','pad_bottom_px']],
            'valid_bounds': bounds, 'measured_pixels': measured_area, 'max_tissue_fraction_224': upper_bound,
            'decision': decision, 'has_effective_nonzero_protein': protein_nonzero,
            'nuclear_markers': effective.loc[effective.nuclear, 'marker_id'].tolist(),
            'body_markers': effective.loc[effective.body, 'marker_id'].tolist(),
            'effective_protein_count': len(selected['protein']), 'nonfinite_effective_pixels': nonfinite,
            'cell_count': len(cells), 'cell_mask_and_single_crop_mapping_match': True,
            'diagnostic_tissue_pixels': tissue_area,
            'diagnostic_tissue_fraction_224': tissue_area/(224*224) if tissue_area is not None else None,
            'diagnostic_tissue_method': tissue_metadata, 'diagnostic_tissue_gap': tissue_gap,
            'original_sampling_status': metadata['sampling']['status'],
            'original_sampling_evaluated': metadata['sampling']['evaluated'],
            'original_sampling_count': metadata['sampling']['count'],
            'figure': str(figure.relative_to(CATALOG))})
        if len(records) % 16 == 0:
            print(f'Checked {len(records)} small images', flush=True)

assert len(records) == 112
for batch in range(math.ceil(len(thumbnails)/16)):
    sheet = Image.new('RGB', (960, 1140), '#12181d')
    for i, tile in enumerate(thumbnails[batch*16:(batch+1)*16]):
        sheet.paste(tile, ((i%4)*240, (i//4)*285))
    sheet.save(OUTPUT / f'contact_{batch+1}.png')
by_study = defaultdict(Counter)
for row in records:
    by_study[row['study']][row['decision']] += 1
summary = {'index_files': index_count, 'indexed_rois': total_count, 'small_rois': len(records),
    'decision_counts': dict(Counter(r['decision'] for r in records)),
    'by_study': {study: dict(counts) for study, counts in by_study.items()},
    'with_canvas_padding': sum(any(r['padding_ltrb']) for r in records),
    'exact_50x50': sum(r['canvas_shape_yx']==[50,50] for r in records),
    'all_effective_signals_finite': all(r['nonfinite_effective_pixels']==0 for r in records),
    'all_have_nonzero_downstream': all(r['has_effective_nonzero_protein'] for r in records),
    'all_cell_mask_crop_mappings_match': all(r['cell_mask_and_single_crop_mapping_match'] for r in records),
    'zero_cell_rois': sum(r['cell_count']==0 for r in records),
    'pending_with_evaluated_true': sum(r['original_sampling_status']=='pending' and r['original_sampling_evaluated'] for r in records),
    'geometrically_capable_rois': [{k:r[k] for k in ['study','roi','figure']}
                                 for r in records if r['max_tissue_fraction_224']>.5],
    'diagnostic_only': True, 'sampling_approved_or_modified': False, 'original_assets_modified': False,
    'html_created_or_modified': False, 'mask_biological_quality_reassessed': False}
with resource_lock(CATALOG):
    registry = json.loads(REGISTRY.read_text())
    actions = {(r['study'],r['roi']):r['action'] for r in registry['decisions']}
    for row in records:
        row['final_user_action'] = actions[row['study'],row['roi']]
    registry['small_images'].update(
        checked=len(records), rois=records, diagnostic_metrics=summary,
        contact_sheets=[str(p.relative_to(CATALOG)) for p in sorted(OUTPUT.glob('contact_*.png'))])
    temporary = REGISTRY.with_suffix('.partial.json')
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(REGISTRY)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
