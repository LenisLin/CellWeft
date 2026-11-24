"""Apply accepted IMC_GC segmentations and remove explicitly approved y padding.

Inputs: private decisions.json exported from the current review and complete products.
Outputs: aligned rectangular products, renewed D01/metadata and decision receipt.
Run --stage first, inspect the exact geometry, then --publish. No model inference.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
from PIL import Image
import tifffile

from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, tiff_source
from cellweft.data.io.writers import write_image, write_json, write_table
from cellweft.data.preparation.crops import attach_crop_origins
from cellweft.data.preparation.indexing import build_cell_table
from cellweft.data.preparation.inputs import (
    assert_reviewed_binding, file_revision, input_binding, load_reviewed_study_inputs,
)
from cellweft.data.preparation.io_coordination import formal_read_lease, formal_write_lease, formal_publish_lease
from cellweft.data.preparation.validation import validate_products

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--decisions', type=Path, required=True)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument('--stage', action='store_true')
mode.add_argument('--publish', action='store_true')
args = parser.parse_args()
job = args.decisions.parent
decisions = json.loads(args.decisions.read_text())
data = Path('/mnt/NAS_21T/ProjectData/CellWeft')
root = data/'processed'/decisions['study']
previous = job.parent
evidence_root = data/'results/imc_gc_user_review_20261119'
evidence_ref = str((evidence_root/'applied_user_decisions.json').relative_to(data))
accepted = {r['image_id']: r for r in decisions['accepted']}
crops = {i: r for i, r in accepted.items() if r['remove_y_padding']}
products = ('channels', 'image', 'mask', 'cells', 'expression', 'neighbors')


def save_pixels(path: Path, pixels: np.ndarray) -> None:
    """Store native scalar pages without changing intensities or interpolation."""
    if pixels.ndim == 2:
        pixels = pixels[None]
    write_image(path, pixels.shape, pixels.dtype,
                lambda c, w: pixels[c, w[1]:w[3], w[0]:w[2]])


if args.stage:
    # Freeze the exact reviewed inputs and retain the previous products on SSD.
    binding = json.loads((previous/'retained_input_binding_958.json').read_text())
    assert_reviewed_binding(root, data_root=data, binding=binding)
    assert not (job/'staged.json').exists()
    tables = ('images.parquet', 'field_mapping.csv', 'review_results.csv',
              'sample_metadata.csv', 'patient_metadata.csv', 'panel_mapping.csv',
              'processing_summary.json', 'README.md')
    backup = job/'backup'
    backup.mkdir()
    for name in tables:
        shutil.copy2(root/name, backup/name)
    write_json(job/'previous_binding.json', binding)
    images = pd.read_parquet(root/'images.parquet')
    images['canvas_policy'] = pd.Series(None, index=images.index, dtype='string')
    fields = pd.read_csv(root/'field_mapping.csv', dtype=str, keep_default_na=False)
    extra = dict.fromkeys(fields.columns, '')
    extra.update(study=decisions['study'], source_file=evidence_ref,
        source_sheet_or_table='accepted', source_field='remove_y_padding',
        target_table='image_metadata', target_field='canvas_policy', join_keys='image_id',
        transformation_rule='Only explicitly approved removal of known zero-valued artificial padding',
        notes=json.dumps({'data_type':'string', 'definition':
            'user_approved_native_rectangle denotes a user-approved native frame with no artificial padding; empty uses the default square policy.'}))
    fields = pd.concat([fields, pd.DataFrame([extra])], ignore_index=True)
    staged = []
    for image_id, decision in accepted.items():
        folder = root/'images'/image_id
        assert decision['source_versions'] == {
            name:file_revision(Path(v['path'])) for name,v in decision['source_versions'].items()}
        if image_id not in crops:
            continue
        local_backup, local = backup/image_id, job/'staged'/image_id
        local_backup.mkdir(); local.mkdir(parents=True)
        with formal_read_lease(folder, data_root=data), resource_lock(root, shared=True), resource_lock(folder, shared=True):
            for path in folder.iterdir():
                if not path.name.startswith('.'):
                    shutil.copy2(path, local_backup/path.name)
        row_index = images.index[images.image_id.eq(image_id)][0]
        row = images.loc[row_index]
        y0, y1 = int(row.pad_top_px), int(row.height_px-row.pad_bottom_px)
        assert y0 > 0 and y1 < row.height_px and row.pad_left_px == row.pad_right_px == 0
        h,w = y1-y0, int(row.width_px)
        for name in ('input.tiff','image.tiff','mask.tiff'):
            pixels = tifffile.imread(local_backup/name)
            assert not np.any(pixels[..., :y0, :]) and not np.any(pixels[..., y1:, :])
            save_pixels(local/name, pixels[..., y0:y1, :])
            assert np.array_equal(tifffile.imread(local/name), pixels[..., y0:y1, :])
        meta = json.loads((local_backup/'metadata.json').read_text())
        mask_identity = uuid.uuid4().hex
        with open_source(tiff_source(local/'mask.tiff', 1)) as reader:
            cells = attach_crop_origins(reader, build_cell_table(reader, decisions['study'], image_id, mask_identity))
        old_cells = pd.read_parquet(local_backup/'cells.parquet')
        assert np.array_equal(cells[['cell_index','mask_label','area']], old_cells[['cell_index','mask_label','area']])
        np.testing.assert_allclose(cells.centroid_y, old_cells.centroid_y-y0, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(cells.centroid_x, old_cells.centroid_x)
        for key in ('bbox_y0','bbox_y1'):
            np.testing.assert_array_equal(cells[key], old_cells[key]-y0)
        write_table(local/'cells.parquet', cells)
        for name in ('channels.parquet','expression.parquet','neighbors.parquet'):
            shutil.copy2(local_backup/name, local/name)
        current = {key: str(local/Path(meta['products'][key]).name) for key in products}
        validate_products(current, (h,w))
        meta['geometry_transform'] = dict(kind='remove_zero_y_padding',
            previous_shape_yx=meta['shape_yx'], previous_window_xyxy=[0,y0,w,y1],
            to_previous_xy={'x_offset':0,'y_offset':y0}, previous_mask_identity=meta['mask_identity'],
            previous_segmentation_task=meta['segmentation_task'],
            previous_versions={p.name:file_revision(p) for p in local_backup.iterdir()},
            raw_source_frame_unchanged=True, native_pixels_removed=0, cells_removed=0,
            normalization_and_expression_reused=True, neighbors_reused=True)
        meta.update(shape_yx=[h,w], mask_identity=mask_identity, segmentation_task=None)
        meta['request']['image']['source']['valid_bounds'] = [0,0,w,h]
        meta['image_request'] = deepcopy(meta['request']['image'])
        write_json(local/'metadata.json', meta)
        images.loc[row_index, ['height_px','pad_top_px','pad_bottom_px','canvas_policy']] = [h,0,0,'user_approved_native_rectangle']
        # A current whole-ROI preview makes the exact retained frame visible.
        channels = pd.read_parquet(local/'channels.parquet')
        with tifffile.TiffFile(local/'image.tiff') as tif:
            groups = [np.mean([tif.pages[int(p)].asarray() for p in
                channels.loc[channels.tiff_page.notna() & channels[key], 'tiff_page']], axis=0)
                for key in ('nuclear','body')]
        nuclear, body = (np.sqrt(np.clip(v,0,1)) for v in groups)
        rgb = np.clip(np.stack([body,np.maximum(nuclear,body*.48),nuclear],axis=-1)*255,0,255).astype('uint8')
        Image.fromarray(rgb).save(local/'cropped_roi.png')
        staged.append(dict(image_id=image_id,case_id=decision['case_id'],previous_shape_yx=meta['geometry_transform']['previous_shape_yx'],
            window_xyxy=[0,y0,w,y1],shape_yx=[h,w],cells=len(cells),removed_cells=0))
    images.to_parquet(job/'images.parquet', index=False)
    fields.to_csv(job/'field_mapping.csv', index=False)
    for image_id in decisions['excluded']:
        assert image_id not in set(images.image_id) and not (root/'images'/image_id).exists()
    write_json(job/'staged.json', {'crops':staged,'accepted_images':list(accepted),'excluded':decisions['excluded'],
        'table_versions':{name:file_revision(root/name) for name in tables},
        'embeddings_version':file_revision(root/'marker_embeddings.pt')})
    print(json.dumps(staged), flush=True)

if args.publish:
    staged = json.loads((job/'staged.json').read_text())
    old_binding = json.loads((job/'previous_binding.json').read_text())
    assert_reviewed_binding(root, data_root=data, binding=old_binding)
    for name, version in staged['table_versions'].items():
        assert file_revision(root/name) == version
    delta = {}
    # Publish only the two transformed images, releasing the NAS writer between them.
    for item in staged['crops']:
        image_id = item['image_id']; folder = root/'images'/image_id
        local, backup = job/'staged'/image_id, job/'backup'/image_id
        with formal_write_lease(folder, data_root=data), resource_lock(root, shared=True), resource_lock(folder,name='.input.lock'), resource_lock(folder,name='.preparation.lock'), resource_lock(folder):
            for name,v in accepted[image_id]['source_versions'].items():
                assert file_revision(folder/name) == v
            meta = json.loads((local/'metadata.json').read_text())
            incomplete = deepcopy(meta); incomplete['completed'] = []
            write_json(folder/'metadata.json', incomplete)
            for name in ('input.tiff','image.tiff','mask.tiff','cells.parquet'):
                target = folder/name
                shutil.copy2(local/name, folder/(name+'.partial'))
                os.replace(folder/(name+'.partial'),target)
                delta[name] = delta.get(name,0) + target.stat().st_size-(backup/name).stat().st_size
            meta['request']['image']['source']['file_identity'] = [file_revision(folder/'input.tiff')]
            meta['image_request'] = deepcopy(meta['request']['image'])
            write_json(local/'metadata.json', meta)
            validate_products(meta['products'], tuple(meta['shape_yx']))
    # Bind the new rectangular input versions and the unchanged remaining scope.
    with formal_publish_lease(root/'images.parquet', directory=root, data_root=data):
        for name in ('images.parquet','field_mapping.csv'):
            shutil.copy2(job/name, root/(name+'.partial'))
            os.replace(root/(name+'.partial'),root/name)
        binding = input_binding(root,data,old_binding['image_ids'])
        reviews = pd.read_csv(root/'review_results.csv', dtype=str, keep_default_na=False)
        d01 = reviews[reviews.check_id.eq('D01')].iloc[-1].to_dict()
        details = json.loads(d01['details'])
        details.update(input_binding=binding, geometry_change={'crops':staged['crops'],
            'authorization':decisions['authorization'], 'removed_native_pixels':0})
        d01.update(preparer='imc_gc_user_decisions_20261120', reviewer='user_20261120',details=json.dumps(details))
        reviews = pd.concat([reviews,pd.DataFrame([d01])],ignore_index=True)
        reviews.to_csv(root/'review_results.csv.partial',index=False)
        os.replace(root/'review_results.csv.partial',root/'review_results.csv')
        assert_reviewed_binding(root,data_root=data,binding=binding)
    write_json(job/'current_binding.json', binding)
    table_versions = {name:file_revision(root/name) for name in
        ('images.parquet','field_mapping.csv','review_results.csv','panel_mapping.csv','sample_metadata.csv')}
    # Refresh complete receipts; all other scientific products remain reusable.
    for number,image_id in enumerate(binding['image_ids'],1):
        folder = root/'images'/image_id
        with formal_write_lease(folder,data_root=data), resource_lock(root,shared=True), resource_lock(folder,name='.preparation.lock'), resource_lock(folder):
            assert all(file_revision(root/name)==v for name,v in table_versions.items())
            metadata_path = folder/'metadata.json'
            before_size = ((job/'backup'/image_id/'metadata.json').stat().st_size if image_id in crops else metadata_path.stat().st_size)
            meta = json.loads(((job/'staged'/image_id/'metadata.json') if image_id in crops else metadata_path).read_text())
            assert meta['input_binding'] == old_binding
            meta['input_binding'] = binding
            meta['input_scope_rebind'] = dict(previous_images=958,current_images=958,
                geometry_changed=image_id in crops, products_recomputed=['cells.crop_origins'] if image_id in crops else [])
            if image_id in accepted:
                meta['segmentation_review'] = dict(status='accepted',reviewer='user',date=decisions['date'],
                    case_id=accepted[image_id]['case_id'], source_versions=accepted[image_id]['source_versions'],
                    geometry_only_update=image_id in crops)
            write_json(metadata_path,meta)
            delta['metadata.json'] = delta.get('metadata.json',0)+metadata_path.stat().st_size-before_size
        if number % 100 == 0:
            print(json.dumps({'rebound':number,'total':len(binding['image_ids'])}),flush=True)
    assert_reviewed_binding(root,data_root=data,binding=binding)
    loaded = load_reviewed_study_inputs(root,data_root=data,image_ids=list(crops),accepted_scope_image_ids=binding['image_ids'])
    assert len(loaded.images)==len(crops)
    for name in ('sample_metadata.csv','patient_metadata.csv'):
        assert (root/name).read_bytes()==(job/'backup'/name).read_bytes()
    assert file_revision(root/'marker_embeddings.pt')==staged['embeddings_version']
    summary = json.loads((root/'processing_summary.json').read_text())
    for name,amount in delta.items():
        summary['file_bytes'][name] += amount
    summary['acceptance']['segmentation_review'] = dict(status='accepted',reviewer='user',date=decisions['date'],
        representative_images=list(accepted),policy='study_stratified_v1')
    summary['acceptance']['user_decisions'] = evidence_ref
    summary['processing']['user_approved_native_rectangles'] = staged['crops']
    summary['nas_five_table_revisions'] = table_versions
    summary['scope']['excluded_input_quality'][0]['user_confirmed'] = True
    with formal_publish_lease(root/'processing_summary.json',directory=root,data_root=data):
        write_json(root/'processing_summary.json',summary)
        readme = (root/'README.md').read_text()
        (root/'README.md').write_text(readme+'\n2026-11-20用户接受8张分层代表图的分割，确认空信号ROI排除。两张代表图仅沿y去掉全零人工补边，保留完整原生采集框、所有细胞及x范围；矩形例外、坐标变换、更新后的mask/index/crop与D01已发布。另1张满幅组织图按用户澄清保持完整。组织估计与训练窗口筛选仍pending，ROI完整保留；具体决定见processing_summary.json的user_decisions引用。\n')
    receipt = dict(study=decisions['study'],date=decisions['date'],status='applied',
        accepted=decisions['accepted'],
        segmentation_accepted=list(accepted),crops=staged['crops'],
        excluded_confirmed=decisions['excluded'],excluded_formal_directory_absent=True,
        retained_complete_images=len(binding['image_ids']),cells=summary['processing']['cells'],
        sampling_reviewed=False,raw_source_files_preserved=True,clinical_tables_unchanged=True,
        marker_embeddings_unchanged=True,evidence=str(args.decisions.resolve()))
    write_json(job/'receipt.json',receipt)
    with formal_write_lease(evidence_root,data_root=data):
        write_json(evidence_root/'applied_user_decisions.json',receipt)
        for item in staged['crops']:
            destination = evidence_root/'crop_updates'/item['case_id']
            destination.mkdir(parents=True,exist_ok=True)
            shutil.copy2(job/'staged'/item['image_id']/'cropped_roi.png',destination/'cropped_roi.png')
    # Keep the adapter's existing per-image export receipts at the accepted geometry.
    images = pd.read_parquet(root/'images.parquet').set_index('image_id')
    for image_id in crops:
        path = previous/'receipts'/f'{image_id}.json'
        old = json.loads(path.read_text())
        old['row'] = json.loads(images.loc[image_id].to_json()) | {'image_id':image_id}
        old['output_version'] = file_revision(root/'images'/image_id/'input.tiff')
        old['user_geometry_decision'] = evidence_ref
        write_json(path,old)
    write_json(previous/'retained_input_binding_958.json',binding)
    write_json(previous/'final_downstream_reception.json',summary)
    print(json.dumps(receipt),flush=True)
