"""Build a private, portable human-review gallery from the completed IMC_GC scope.

Inputs: current image/mask/channel assets and the existing stratified review.
Outputs: standalone HTML, same-coordinate PNGs and a decision template in results.
Formal scientific products are read-only; user selections are exported for later application.
"""
from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import runpy
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter, minimum_filter
import tifffile

from cellweft.data.preparation.crops import grid_origins
from cellweft.data.preparation.inputs import file_revision
from cellweft.data.preparation.io_coordination import formal_read_lease, formal_write_lease
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.writers import write_json

PROJECT = Path(__file__).resolve().parents[3]
DATA = Path('/mnt/NAS_21T/ProjectData/CellWeft')
ROOT = DATA / 'processed/IMC_GC'
JOB = PROJECT / 'cache/imc_gc_input_20261119'
OUTPUT = JOB / 'user_review'
DESTINATION = DATA / 'results/imc_gc_user_review_20261119'
OUTPUT.mkdir(parents=True, exist_ok=True)
helpers = runpy.run_path(str(Path(__file__).with_name('01_generate_html.py')))
colorize_mask = helpers['colorize_mask']
review = json.loads((JOB / 'downstream_representative_review.json').read_text())
images = pd.read_parquet(ROOT / 'images.parquet').set_index('image_id')
summary = json.loads((ROOT / 'processing_summary.json').read_text())
tissue = {row['image_id']: row for row in review['tissue_sampling_review']['estimates']}
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)


def png_uri(image: Image.Image) -> str:
    """Embed review pixels so the HTML remains portable without a web server."""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def signal_rgb(nuclear: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Use fixed sqrt display enhancement; cyan DNA and orange body share geometry."""
    n, b = np.sqrt(np.clip(nuclear, 0, 1)), np.sqrt(np.clip(body, 0, 1))
    rgb = np.stack([b, np.maximum(n, b * .48), n], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def cell_edges(mask: np.ndarray) -> np.ndarray:
    """Retain both instance-to-instance and instance-to-background borders."""
    return ((maximum_filter(mask, size=3) != minimum_filter(mask, size=3)) & (mask > 0))


def render_case(image_id: str, number: int) -> dict:
    """Render native windows from actual shared channels and the final mask."""
    folder = ROOT / 'images' / image_id
    local = OUTPUT / 'staging'
    local.mkdir(exist_ok=True)
    with formal_read_lease(folder, data_root=DATA), resource_lock(ROOT, shared=True), resource_lock(folder, shared=True):
        versions = {name: file_revision(folder / name) for name in
                    ('image.tiff', 'mask.tiff', 'channels.parquet', 'cells.parquet', 'metadata.json')}
        for name in versions:
            shutil.copy2(folder / name, local / name)
        assert versions == {name: file_revision(folder / name) for name in versions}
    channels = pd.read_parquet(local / 'channels.parquet')
    active = channels[channels.tiff_page.notna()]
    nuclear_rows, body_rows = active[active.nuclear], active[active.body]
    with tifffile.TiffFile(local / 'image.tiff') as tif:
        nuclear = np.mean([tif.pages[int(p)].asarray() for p in nuclear_rows.tiff_page], axis=0)
        body = np.mean([tif.pages[int(p)].asarray() for p in body_rows.tiff_page], axis=0)
    mask = tifffile.imread(local / 'mask.tiff')
    cells = pd.read_parquet(local / 'cells.parquet')
    metadata = json.loads((local / 'metadata.json').read_text())
    h, w = mask.shape
    origins = grid_origins((h, w))
    # A stresses crowded cells; B is a reproducible nonempty, spatially distinct grid.
    counts = [int(((cells.centroid_x >= x) & (cells.centroid_x < x + 224) &
                   (cells.centroid_y >= y) & (cells.centroid_y < y + 224)).sum()) for x, y in origins]
    dense = origins[int(np.argmax(counts))]
    options = [origin for origin, count in zip(origins, counts) if count and
               (abs(origin[0] - dense[0]) >= 224 or abs(origin[1] - dense[1]) >= 224)]
    random_origin = options[int(np.random.default_rng(20261119 + number).integers(len(options)))] if options else dense
    windows = [dense, random_origin]
    rgb, edges = signal_rgb(nuclear, body), cell_edges(mask)
    case_dir = OUTPUT / f'R{number:02d}'
    case_dir.mkdir(exist_ok=True)
    overview = Image.fromarray(rgb)
    overlay = Image.fromarray(np.dstack([np.full_like(mask, 255, dtype=np.uint8)] * 3 + [edges.astype(np.uint8) * 255]), 'RGBA')
    for image, resample in ((overview, Image.Resampling.BILINEAR), (overlay, Image.Resampling.NEAREST)):
        image.thumbnail((1000, 1000), resample=resample)
    draw = ImageDraw.Draw(overview)
    scale = overview.width / w
    for label, (x, y), color in zip(('A', 'B'), windows, ('#ff7777', '#70baff')):
        draw.rectangle((x * scale, y * scale, min(x + 224, w) * scale, min(y + 224, h) * scale), outline=color, width=2)
        draw.text((x * scale + 4, y * scale + 3), label, fill=color, font=FONT)
    overview.save(case_dir / 'overview.png')
    overlay.save(case_dir / 'edges.png')
    board = Image.new('RGB', (4 * 460 + 12, 2 * 510 + 28), '#111820')
    pen = ImageDraw.Draw(board)
    for row_index, (x, y) in enumerate(windows):
        ys, xs = slice(y, min(y + 224, h)), slice(x, min(x + 224, w))
        label_mask = mask[ys, xs]
        composite = rgb[ys, xs].copy()
        composite[edges[ys, xs]] = 255
        panels = [np.repeat((np.sqrt(np.clip(nuclear[ys, xs], 0, 1))*255).astype(np.uint8)[..., None], 3, axis=2),
                  np.repeat((np.sqrt(np.clip(body[ys, xs], 0, 1))*255).astype(np.uint8)[..., None], 3, axis=2),
                  composite, colorize_mask(label_mask)]
        for col, (title, pixels) in enumerate(zip(('DNA mean', 'Body mean', 'Signal + cell borders', 'Cell instances'), panels)):
            tile = Image.fromarray(pixels).resize((448, 448), Image.Resampling.NEAREST)
            left, top = 12 + col * 460, 12 + row_index * 510
            pen.text((left, top), f'{"AB"[row_index]} | {title}', fill='white', font=FONT)
            pen.text((left, top + 20), f'x=[{x},{min(x+224,w)}) y=[{y},{min(y+224,h)})', fill='#c9d5df', font=FONT)
            board.paste(tile, (left, top + 44))
    board.save(case_dir / 'local_windows.png')
    row = images.loc[image_id]
    pixel_size = None if pd.isna(row.pixel_size_x_um) else float(row.pixel_size_x_um)
    record = {'case_id': f'R{number:02d}', 'image_id': image_id, 'panel_id': row.panel_id,
              'kind': 'retained', 'priority': '组织范围优先确认' if number <= 2 else '跨panel补充抽样',
              'shape_yx': [h, w], 'cell_count': len(cells), 'pixel_size_um': pixel_size,
              'windows_xyxy': [[x, y, min(x+224,w), min(y+224,h)] for x, y in windows],
              'window_selection': ['densest centroid grid', 'fixed-seed nonempty separated grid'],
              'nuclear_markers': nuclear_rows.marker_id.tolist(), 'body_markers': body_rows.marker_id.tolist(),
              'source_versions': versions,
              'segmentation_decision': 'accept' if metadata.get('segmentation_review', {}).get('status') == 'accepted' else 'pending',
              'tissue_decision': 'pending', 'notes': '',
              'overview': png_uri(overview), 'edges': png_uri(overlay), 'windows': png_uri(board)}
    if image_id in tissue and 'geometry_transform' not in metadata:
        path = JOB / 'downstream_run/overviews' / (image_id + '_tissue.png')
        shutil.copy2(path, case_dir / 'tissue_estimate.png')
        with Image.open(path) as image:
            assert image.size == (w, h)
            record['tissue_image'] = png_uri(image)
        record['tissue_estimate'] = tissue[image_id]
    shutil.rmtree(local)
    return record


# Reuse the exact 8 reviewed representatives; the first two are the panel_02 concerns.
records = [render_case(image_id, number) for number, image_id in enumerate(review['image_ids'], 1)]
for excluded in summary['scope']['excluded_input_quality']:
    with Image.open(JOB / 'empty_mask_review/dna_body_signal_overview.png') as image:
        records.append({'case_id': f'R{len(records)+1:02d}', 'image_id': excluded['image_id'],
                        'kind': 'excluded', 'priority': '已排除ROI的处置复核', 'reason': excluded['reason'],
                        'overview': png_uri(image),
                        'segmentation_decision': 'confirm_exclusion' if excluded.get('user_confirmed') else 'pending', 'notes': ''})

template = Path(__file__).with_name('imc_gc_review_template.html').read_text()
payload = json.dumps(records, ensure_ascii=False).replace('</', '<\\/')
(OUTPUT / 'index.html').write_text(template.replace('__REVIEW_DATA__', payload))
manifest = {'study': 'IMC_GC', 'retained_scope': 958, 'review_policy': 'existing 8 stratified representatives + excluded empty field',
            'display': {'signal_gamma': .5, 'local_window_px': 224, 'local_display_scale': 2},
            'cases': [{k: v for k, v in record.items() if k not in {'overview','edges','windows','tissue_image'}} for record in records]}
write_json(OUTPUT / 'review_manifest.json', manifest)
(OUTPUT / 'README.md').write_text('# IMC_GC人工确认包\n\n打开index.html。R01–R02优先确认组织估计与分割，R03–R08为跨panel补充抽样，R09复核已排除的空信号图。\n\n青色为DNA均值，橙色为所选五个body通道的有效均值，白线为当前细胞边界；预览统一γ=0.5增强弱信号。局部A为最密集网格，B为固定种子随机非空网格，224×224原生像素以2倍最近邻展示；未知µm尺度保持未知。绿色组织估计与细胞分割分别确认。\n\n页面可导出判断JSON，也可直接按R01–R09在对话回复；页面选择仅记录反馈，不修改正式产物。\n')
with formal_write_lease(DESTINATION, data_root=DATA):
    shutil.copytree(OUTPUT, DESTINATION, dirs_exist_ok=True)
print(json.dumps({'cases': len(records), 'html': str(DESTINATION / 'index.html'), 'bytes': (DESTINATION / 'index.html').stat().st_size}), flush=True)
