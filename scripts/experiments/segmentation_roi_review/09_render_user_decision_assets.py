"""Render marker exclusions, native-size small FOVs and a candidate comparison.

Inputs: the checked user decisions, current channel/mask assets and candidate mask.
Outputs: PNG panels and an asset index in user_decisions_20261119; no cell tables.
"""

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont


ROOT = Path('/mnt/NAS_21T/ProjectData/CellWeft/processed')
REPORT = ROOT / '_catalog/public/evidence/segmentation_followup_20261119'
OUTPUT = REPORT / 'user_decisions_20261119'
helpers = runpy.run_path(str(Path(__file__).with_name('01_generate_html.py')))
read_windows = helpers['tiff_windows']
colorize = helpers['colorize_mask']
decisions = json.loads((OUTPUT / 'decisions.json').read_text())
review = json.loads((REPORT / 'review_summary.json').read_text())
records = {(r['study'], r['roi']): r for r in review['rois']}
assets = {}


def panels_for(study, roi, bounds, excluded):
    """Read matched pixels for up to five effective segmentation channels."""
    source = ROOT / study / 'images' / roi
    table = pd.read_parquet(source / 'channels.parquet')
    selected = table[table.keep_channel & table.tiff_page.notna()
                     & ~table.marker_id.isin(excluded)]
    nuclear = selected[selected.nuclear].sort_values('tiff_page').head(2)
    body = selected[selected.body & ~selected.nuclear].sort_values('tiff_page').head(5-len(nuclear))
    selected = pd.concat([nuclear, body])
    assert 3 <= len(selected) <= 5
    panels, nucleus, cytoplasm = [], [], []
    with tifffile.TiffFile(source / 'image.tiff') as image:
        for row in selected.to_dict('records'):
            values = read_windows(image.pages[int(row['tiff_page'])], [bounds])[0].astype(np.float32)
            scale = helpers['display_scale'](values, row['q995'])
            values = np.clip(values / scale, 0, 1)
            (nucleus if row['nuclear'] else cytoplasm).append(values)
            panels.append((row['marker_id'], Image.fromarray((values*255).astype(np.uint8))))
    with tifffile.TiffFile(source / 'mask.tiff') as image:
        mask = read_windows(image.pages[0], [bounds])[0]
    composite = np.zeros(mask.shape + (3,), dtype=np.uint8)
    composite[..., 0] = (np.maximum.reduce(cytoplasm)*255).astype(np.uint8)
    composite[..., 1] = (np.maximum.reduce(cytoplasm)*145).astype(np.uint8)
    composite[..., 2] = (np.maximum.reduce(nucleus)*255).astype(np.uint8)
    panels += [('Current mask', Image.fromarray(colorize(mask))),
               ('Signal + current contours', Image.fromarray(contours(composite, mask)))]
    return panels, composite


def contours(composite, mask):
    """Draw instance edges at the same XY, without changing the underlying data."""
    edges = np.zeros(mask.shape, dtype=bool)
    edges[1:] |= mask[1:] != mask[:-1]
    edges[:-1] |= mask[1:] != mask[:-1]
    edges[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edges[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    result = composite.copy()
    result[edges] = 255
    return result


def board(panels, title, path):
    """Keep each 512-pixel crop at one pixel per output pixel."""
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    width = panels[0][1].width
    cols = 3
    canvas = Image.new('RGB', (cols*(width+16)+16,
                              ((len(panels)+cols-1)//cols)*(width+40)+66), '#12181d')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), title, font=font, fill='white')
    for i, (label, image) in enumerate(panels):
        x, y = 16+(i%cols)*(width+16), 55+(i//cols)*(width+40)
        draw.text((x, y), label, font=font, fill='white')
        canvas.paste(image.convert('RGB'), (x, y+28))
    canvas.save(path)


for row in decisions['decisions']:
    if row['action'] != 'exclude_channels':
        continue
    study, roi = row['study'], row['roi']
    bounds = records[study, roi]['window_xyxy']
    folder = OUTPUT / 'figures' / roi
    folder.mkdir(parents=True, exist_ok=True)
    panels, _ = panels_for(study, roi, bounds, row['channels'])
    path = folder / 'effective_channels.png'
    board(panels, roi + ' | excluded: ' + ', '.join(row['channels']), path)
    assets[roi] = {'kind': 'excluded_channels', 'figure': str(path.relative_to(REPORT)),
                   'channels': [label for label, _ in panels[:-2]], 'window_xyxy': bounds}

for roi in decisions['display_query_rois']:
    study = 'Lyman2025_IMC_PanIN'
    folder = OUTPUT / 'figures' / roi
    folder.mkdir(parents=True, exist_ok=True)
    panels, _ = panels_for(study, roi, [0, 0, 50, 50], [])
    native = []
    for i, (label, image) in enumerate(panels):
        path = folder / f'native_{i}.png'
        assert image.size == (50, 50)
        image.save(path)
        native.append({'label': label, 'path': str(path.relative_to(REPORT))})
    assets[roi] = {'kind': 'native_small_fov', 'shape_yx': [50, 50],
                   'pixel_size_um': 1.0, 'panels': native}

candidate = OUTPUT / 'INT_Donor_E_candidate/mask.tiff'
if candidate.is_file():
    study, roi = 'Allam2024_IMC_NormalSLO', 'INT_Donor_E'
    row = records[study, roi]
    windows = [row['window_xyxy'], *row['previous_random_windows_xyxy']]
    figures = []
    with tifffile.TiffFile(candidate) as tif:
        assert tif.pages[0].shape == (5000, 5000)
        masks = read_windows(tif.pages[0], windows)
    for i, (bounds, mask) in enumerate(zip(windows, masks)):
        panels, composite = panels_for(study, roi, bounds, [])
        panels.extend([('Candidate mask (not accepted)', Image.fromarray(colorize(mask))),
                       ('Signal + candidate contours', Image.fromarray(contours(composite, mask)))])
        path = candidate.parent / f'comparison_{i+1}.png'
        board(panels, f'{roi} | x=[{bounds[0]},{bounds[2]}) y=[{bounds[1]},{bounds[3]})', path)
        figures.append(str(path.relative_to(REPORT)))
    assets[roi] = {'kind': 'resegmentation_candidate', 'figures': figures,
                   'status': 'candidate_only_not_published', 'windows_xyxy': windows}

(OUTPUT / 'display_assets.json').write_text(json.dumps(assets, ensure_ascii=False, indent=2)+'\n')
print(f'Rendered {len(assets)} ROI asset groups')
