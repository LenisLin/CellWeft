"""Behavioral contracts for native-window preparation and dependency recovery."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from cellweft.data.contracts import ImageRecord, MaskSource, PreparationSettings, SourceSpec
from cellweft.data.io.readers import open_source, tiff_source
from cellweft.data.io.writers import write_image, write_json, write_table
from cellweft.data.preparation.crops import attach_crop_origins, grid_origins, read_crop
from cellweft.data.preparation.indexing import build_cell_table
from cellweft.data.preparation.intensity import channel_percentile
from cellweft.data.preparation.markers import build_marker_registry, build_panel_mapping
from cellweft.data.preparation.masks import check_existing_mask
from cellweft.data.preparation.neighbors import compute_neighbors
from cellweft.data.preparation.pipeline import prepare_image
from cellweft.data.preparation.segmentation import stitch_instances


def save_tiff(path, pixels, tile=16):
    if pixels.ndim == 2:
        pixels = pixels[None]
    with tifffile.TiffWriter(path) as writer:
        for plane in pixels:
            writer.write(plane,tile=(tile,tile),photometric='minisblack',metadata=None,compression='deflate')


def panel(count):
    rows = []
    for i in range(count):
        rows.append(dict(panel_id='p',source_channel_index=i,source_name=f'P{i}',marker_id=f'p{i}',
                         role='downstream',nuclear=i==0,body=i==1,exclusion_reason=''))
    return pd.DataFrame(rows)


def setup_case(tmp_path, pixels=None, labels=None):
    if pixels is None:
        y,x = np.indices((37,49))
        pixels = np.stack([x+y,x*2+y]).astype(np.float32)
    raw = tmp_path/'raw.tiff';save_tiff(raw,pixels)
    mask = None
    if labels is not None:
        mask_path = tmp_path/'supplied.tiff';save_tiff(mask_path,labels)
        mask = MaskSource(mask_path,'whole_cell')
    source = SourceSpec('image','p','tiff',raw,tuple(range(len(pixels))),mask=mask)
    record = ImageRecord('study','image',len(pixels),(.5,.5),tmp_path,tmp_path/'panel.csv',tmp_path/'processed')
    settings = PreparationSettings(tmp_path/'cache',tile_size=16,block_size=19)
    return source,record,settings,panel(len(pixels))


def run_case(source,record,settings,mapping,executor=None):
    with open_source(source) as reader:
        return prepare_image(record,source,reader,mapping,settings,executor)


def test_tiff_tile_windows_and_source_roi(tmp_path):
    pixels=np.arange(3*37*49,dtype=np.uint16).reshape(3,37,49)
    source,_,_,_=setup_case(tmp_path,pixels)
    source=replace(source,window=(7,4,45,35))
    with open_source(source) as reader:
        assert reader.shape==(3,31,38)
        np.testing.assert_array_equal(reader.read_window([2,0],(3,2,30,23)),pixels[[2,0],6:27,10:37])
        rebuilt=np.zeros(reader.shape,dtype=reader.dtype)
        for block in reader.iter_blocks([0,1,2],11):
            x,y,xx,yy=block.window;rebuilt[:,y:yy,x:xx]=block.values
        np.testing.assert_array_equal(rebuilt,pixels[:,4:35,7:45])


def test_stripped_tiff_window(tmp_path):
    pixels=np.arange(47*61,dtype=np.float32).reshape(47,61)
    path=tmp_path/'strips.tiff';tifffile.imwrite(path,pixels,rowsperstrip=7,compression='deflate')
    with open_source(tiff_source(path,1)) as reader:
        np.testing.assert_array_equal(reader.read_window([0],(9,3,41,37))[0],pixels[3:37,9:41])


def test_exact_quantile_with_invalid_domain_and_measured_zeros(tmp_path):
    rng=np.random.default_rng(9)
    pixels=rng.normal(size=(1,37,49)).astype(np.float32);pixels[:,:,::2]=0
    source,_,settings,_=setup_case(tmp_path,pixels)
    valid=np.ones((37,49),np.uint8);valid[:3]=0
    validity=tmp_path/'valid.tiff';save_tiff(validity,valid)
    source=replace(source,valid_mask=validity)
    with open_source(source) as reader:
        q,reason=channel_percentile(reader,0,settings.cache_dir,13)
    assert not reason
    assert q==pytest.approx(np.percentile(pixels[0][valid>0].astype(np.float64),99.5,method='linear'))


def test_pipeline_identity_means_and_resume(tmp_path):
    labels=np.zeros((37,49),np.uint32);labels[2:13,3:21]=17;labels[18:35,20:47]=2_000_000_000
    source,record,settings,mapping=setup_case(tmp_path,labels=labels)
    result=run_case(source,record,settings,mapping)
    cells=pd.read_parquet(result.products['cells']);expr=pd.read_parquet(result.products['expression'])
    assert list(cells.mask_label)==[17,2_000_000_000]
    assert list(cells.cell_index)==[0,1]
    with open_source(tiff_source(Path(result.products['image']),2)) as reader:
        full=reader.read_window([0,1],(0,0,49,37))
        for i,label in enumerate(cells.mask_label):
            np.testing.assert_allclose(expr.loc[i,['p0','p1']].to_numpy(dtype=float),full[:,labels==label].mean(axis=1,dtype=np.float64),rtol=1e-6)
        with open_source(tiff_source(Path(result.products['mask']),1)) as mask:
            crop,cm,shape=read_crop(reader,mask,(0,0),[0,1])
            assert crop.shape==(2,224,224) and shape==(37,49)
            np.testing.assert_array_equal(cm[:37,:49],labels)
    assert run_case(source,record,settings,mapping).reused
    image_mtime=Path(result.products['image']).stat().st_mtime_ns
    old_ids=cells.cell_id.tolist()
    labels[2:13,3:21]=101;save_tiff(source.mask.path,labels)
    refreshed=run_case(source,record,settings,mapping)
    assert Path(refreshed.products['image']).stat().st_mtime_ns==image_mtime
    assert pd.read_parquet(refreshed.products['cells']).cell_id.tolist()!=old_ids
    assert list(pd.read_parquet(refreshed.products['cells']).mask_label)==[101,2_000_000_000]


def test_channel_exclusions_are_per_channel(tmp_path):
    pixels=np.zeros((4,37,49),np.float32)
    pixels[0]=np.arange(37*49).reshape(37,49)
    pixels[1,0,0]=1000
    pixels[2]=3
    pixels[3,0,0]=np.nan
    labels=np.zeros((37,49),np.uint32);labels[1:8,1:8]=9
    source,record,settings,mapping=setup_case(tmp_path,pixels,labels)
    result=run_case(source,record,settings,mapping)
    channels=pd.read_parquet(result.products['channels'])
    assert len(channels)==4
    assert channels.loc[0,'tiff_page']==0 and channels.loc[2,'tiff_page']==1
    assert pd.isna(channels.loc[1,'tiff_page']) and pd.isna(channels.loc[3,'tiff_page'])
    assert 'nonfinite' in ' '.join(result.gaps)
    expression=pd.read_parquet(result.products['expression'])
    assert expression.columns.tolist()==['cell_index','mask_label','p0','p2']
    assert expression.p2.iloc[0]==1


def test_mask_without_nuclear_or_any_protein(tmp_path):
    pixels=np.zeros((2,37,49),np.float32)
    labels=np.zeros((37,49),np.uint32);labels[2:9,3:11]=99
    source,record,settings,mapping=setup_case(tmp_path,pixels,labels)
    result=run_case(source,record,settings,mapping)
    assert {'cells','mask','neighbors'}<=result.products.keys()
    assert 'image' not in result.products and 'expression' not in result.products


def test_existing_mask_four_checks(tmp_path):
    labels=np.zeros((37,49),np.uint32);labels[3:8,2:9]=12
    path=tmp_path/'labels.tiff';save_tiff(path,labels)
    with open_source(tiff_source(path,1)) as reader:
        assert check_existing_mask(reader,(37,49),MaskSource(path,'whole_cell'))['usable']
        assert not check_existing_mask(reader,(37,49),MaskSource(path,'nuclear'))['usable']
        assert not check_existing_mask(reader,(38,49),MaskSource(path,'whole_cell'))['usable']


def test_actual_crop_intersections_and_shared_indices(tmp_path):
    labels=np.zeros((337,449),np.uint32)
    labels[5:330,5]=17;labels[329,5:440]=17  # Bounding box includes windows without this L-shaped cell.
    labels[150:160,150:160]=1000
    path=tmp_path/'labels.tiff';save_tiff(path,labels)
    with open_source(tiff_source(path,1)) as reader:
        cells=attach_crop_origins(reader,build_cell_table(reader,'s','i','mask'))
    for row in cells.itertuples():
        expected=[{'x0':x,'y0':y} for x,y in grid_origins(labels.shape) if np.any(labels[y:y+224,x:x+224]==row.mask_label)]
        assert row.crop_origins==expected
    for n in [1,223,224,225,336,337,449]:
        origins=grid_origins((n,n))
        covered=np.zeros((n,n),bool)
        for x,y in origins:covered[y:y+224,x:x+224]=True
        assert covered.all() and len(origins)==len(set(origins))


def test_neighbors_coincident_and_small_counts():
    for n in [0,1,2,25]:
        cells=pd.DataFrame({'cell_index':np.arange(n),'centroid_x':np.zeros(n),'centroid_y':np.zeros(n)})
        result=compute_neighbors(cells)
        assert result.shape==(n,21)
        for i,row in enumerate(result.iloc[:,1:].to_numpy()):
            assert i not in row
            assert np.sum(row>=0)==min(20,n-1)


def test_empty_cell_table_serialization(tmp_path):
    path=tmp_path/'mask.tiff';save_tiff(path,np.zeros((17,23),np.uint32))
    with open_source(tiff_source(path,1)) as reader:
        cells=attach_crop_origins(reader,build_cell_table(reader,'s','i','m'))
    write_table(tmp_path/'cells.parquet',cells)
    assert pd.read_parquet(tmp_path/'cells.parquet').empty


def test_pending_segmentation_file_handoff(tmp_path):
    source,record,settings,mapping=setup_case(tmp_path)
    weights=tmp_path/'weights';weights.write_text('test identity, not a real model')
    settings=replace(settings,weights=weights)
    first=run_case(source,record,settings,mapping)
    task=json.loads(first.metadata_path.read_text())['segmentation_task']
    assert task and 'mask' not in first.products
    labels=np.zeros((37,49),np.uint32);labels[2:10,2:10]=40
    result_path=Path(task['result_path']);save_tiff(result_path.parent/'mask.tiff',labels)
    write_json(result_path,{'task_id':task['task_id'],'request':task['request'],
                           'parameters':task['parameters'],'stitching':task['stitching'],
                           'content_identity':task['content_identity'],
                           'mask':'mask.tiff','whole_cell_confirmed':True})
    resumed=run_case(source,record,settings,mapping)
    assert 'expression' in resumed.products and 'mask' in resumed.products
    assert run_case(source,record,settings,mapping).reused


def test_stale_segmentation_result_rejected(tmp_path):
    source,record,settings,mapping=setup_case(tmp_path)
    weights=tmp_path/'weights';weights.write_text('fixture')
    settings=replace(settings,weights=weights)
    first=run_case(source,record,settings,mapping)
    task=json.loads(first.metadata_path.read_text())['segmentation_task']
    write_json(Path(task['result_path']),{'task_id':'stale','request':task['request']})
    with pytest.raises(ValueError,match='does not match'):
        run_case(source,record,settings,mapping)


def test_stitching_cross_core_instances(tmp_path):
    from scipy.ndimage import label
    signal=np.zeros((2,100,120),np.float32)
    signal[:,20:80,25:95]=1
    path=tmp_path/'signal.tiff';save_tiff(path,signal)
    with open_source(tiff_source(path,2)) as reader:
        stitch_instances(reader,lambda x:label(x[:,:,1]>0)[0],tmp_path,tmp_path/'mask.tiff',
                         core_size=40,halo=16,tile_size=16)
    with open_source(tiff_source(tmp_path/'mask.tiff',1)) as reader:
        mask=reader.read_window([0],(0,0,120,100))[0]
    assert len(np.unique(mask[mask>0]))==1
    np.testing.assert_array_equal(mask>0,signal[0]>0)


def test_registry_family_and_duplicate_channel_mapping():
    registry=build_marker_registry(pd.DataFrame([
        dict(marker_id='family',display_name='panCK',canonical_protein_id=None,measured_form='family')]))
    mapping=panel(1);mapping['marker_id']='family'
    assert 'p' in build_panel_mapping(mapping,registry)
    doubled=pd.concat([mapping,mapping.assign(source_channel_index=1)],ignore_index=True)
    with pytest.raises(ValueError,match='duplicate'):
        build_panel_mapping(doubled,registry)


def test_zarr_source_is_read_only_and_roi_local(tmp_path):
    zarr=pytest.importorskip('zarr')
    pixels=np.arange(2*17*23,dtype=np.float32).reshape(2,17,23)
    path=tmp_path/'raw.zarr'
    array=zarr.open_array(str(path),mode='w',shape=pixels.shape,chunks=(1,8,8),dtype='float32')
    array[:]=pixels
    source=SourceSpec('i','p','zarr',path,(1,0),window=(3,2,20,16),source_revision='fixture')
    with open_source(source) as reader:
        np.testing.assert_array_equal(reader.read_window([0,1],(0,0,17,14)),pixels[[1,0],2:16,3:20])


def test_ims_source_explicit_z_and_valid_shape(tmp_path):
    h5py=pytest.importorskip('h5py')
    path=tmp_path/'image.ims'
    pixels=np.arange(2*17*23,dtype=np.float32).reshape(2,17,23)
    with h5py.File(path,'w') as handle:
        handle.create_dataset('DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data',data=pixels)
    source=SourceSpec('i','p','ims',path,(0,),z_index=1,valid_shape=(15,20))
    with open_source(source) as reader:
        assert reader.shape==(1,15,20)
        np.testing.assert_array_equal(reader.read_window([0],(2,3,10,11))[0],pixels[1,3:11,2:10])


def test_mask_and_panel_changes_invalidate_only_dependencies(tmp_path):
    labels=np.zeros((37,49),np.uint32);labels[2:10,3:12]=7
    source,record,settings,mapping=setup_case(tmp_path,labels=labels)
    first=run_case(source,record,settings,mapping)
    mask_id=json.loads(first.metadata_path.read_text())['mask_identity']
    altered=mapping.copy();altered.loc[0,'marker_id']='renamed'
    second=run_case(source,record,settings,altered)
    assert 'renamed' in pd.read_parquet(second.products['expression'])
    assert json.loads(second.metadata_path.read_text())['mask_identity']==mask_id
    # An interrupted update without completion metadata must be reconstructed.
    second.metadata_path.unlink()
    third=run_case(source,record,settings,altered)
    assert not third.reused and third.metadata_path.is_file()


def complete_fixture_task(result, labels):
    task=json.loads(result.metadata_path.read_text())['segmentation_task']
    path=Path(task['result_path'])
    save_tiff(path.parent/'mask.tiff',labels)
    write_json(path,{'task_id':task['task_id'],'request':task['request'],
                     'parameters':task['parameters'],'stitching':task['stitching'],
                     'content_identity':task['content_identity'],
                     'mask':'mask.tiff','whole_cell_confirmed':True})
    return task


def test_predicted_mask_invalidates_when_rejected_source_exists(tmp_path):
    labels=np.zeros((37,49),np.uint32);labels[3:12,4:15]=5
    source,record,settings,mapping=setup_case(tmp_path,labels=labels)
    source=replace(source,mask=replace(source.mask,mask_type='nuclear'))
    weights=tmp_path/'weights';weights.write_text('fixture')
    settings=replace(settings,weights=weights)
    first=run_case(source,record,settings,mapping)
    task=complete_fixture_task(first,labels)
    second=run_case(source,record,settings,mapping)
    assert json.loads(second.metadata_path.read_text())['mask_origin']=='cpsam_v2'
    changed=replace(settings,segmentation_parameters={**settings.segmentation_parameters,'flow_threshold':0.8})
    third=run_case(source,record,changed,mapping)
    metadata=json.loads(third.metadata_path.read_text())
    assert 'mask' not in third.products
    assert metadata['segmentation_task']['task_id']!=task['task_id']
    assert not metadata['source_mask_check']['usable']


def test_source_mask_bound_to_spatial_selection(tmp_path):
    labels=np.ones((37,49),np.uint32)
    source,record,settings,mapping=setup_case(tmp_path,labels=labels)
    source=replace(source,window=(0,0,20,20),mask=replace(source.mask,window=(0,0,20,20)))
    first=run_case(source,record,settings,mapping)
    identity=json.loads(first.metadata_path.read_text())['mask_identity']
    # Same dimensions still require a fresh coarse source check for the new frame.
    source=replace(source,window=(5,5,25,25))
    second=run_case(source,record,settings,mapping)
    assert json.loads(second.metadata_path.read_text())['mask_identity']!=identity


def test_overview_rejection_and_required_judgment(tmp_path):
    labels=np.ones((37,49),np.uint32)
    source,record,settings,mapping=setup_case(tmp_path,labels=labels)
    with open_source(tiff_source(source.mask.path,1)) as reader:
        assert not check_existing_mask(reader,(37,49),replace(source.mask,overview_judgment=False))['usable']
        assert not check_existing_mask(reader,(37,49),replace(source.mask,overview_required=True))['usable']
        assert check_existing_mask(reader,(37,49),replace(source.mask,overview_required=True,overview_judgment=True))['usable']
    rejected=replace(source,mask=replace(source.mask,overview_judgment=False))
    assert 'mask' not in run_case(rejected,record,settings,mapping).products


def test_missing_expression_recovers_without_segmentation_cache(tmp_path):
    import shutil
    source,record,settings,mapping=setup_case(tmp_path)
    weights=tmp_path/'weights';weights.write_text('fixture')
    settings=replace(settings,weights=weights)
    first=run_case(source,record,settings,mapping)
    labels=np.ones((37,49),np.uint32)
    task=complete_fixture_task(first,labels)
    second=run_case(source,record,settings,mapping)
    identity=json.loads(second.metadata_path.read_text())['mask_identity']
    image_mtime=Path(second.products['image']).stat().st_mtime_ns
    Path(second.products['expression']).unlink()
    shutil.rmtree(Path(task['input_path']).parent)
    third=run_case(source,record,settings,mapping)
    assert Path(third.products['expression']).is_file()
    assert json.loads(third.metadata_path.read_text())['mask_identity']==identity
    assert Path(third.products['image']).stat().st_mtime_ns==image_mtime


def test_missing_pending_input_is_rebuilt(tmp_path):
    source,record,settings,mapping=setup_case(tmp_path)
    weights=tmp_path/'weights';weights.write_text('fixture')
    settings=replace(settings,weights=weights)
    first=run_case(source,record,settings,mapping)
    task=json.loads(first.metadata_path.read_text())['segmentation_task']
    Path(task['input_path']).unlink()
    second=run_case(source,record,settings,mapping)
    new_task=json.loads(second.metadata_path.read_text())['segmentation_task']
    assert new_task['task_id']!=task['task_id'] and Path(new_task['input_path']).is_file()


def test_transported_inputs_checked_before_model_loading(tmp_path):
    from cellweft.data.preparation.segmentation import CellposeExecutor, verify_task_inputs
    source,record,settings,mapping=setup_case(tmp_path)
    weights=tmp_path/'weights';weights.write_bytes(b'checkpoint_a')
    settings=replace(settings,weights=weights)
    first=run_case(source,record,settings,mapping)
    task=json.loads(first.metadata_path.read_text())['segmentation_task']
    alternate=tmp_path/'other_weights';alternate.write_bytes(b'checkpoint_b')
    with pytest.raises(ValueError,match='content differs'):
        CellposeExecutor().execute({**task,'weights':str(alternate)})
    wrong_signal=tmp_path/'wrong_signal.tiff';save_tiff(wrong_signal,np.ones((2,37,49),np.float32))
    with pytest.raises(ValueError,match='content differs'):
        CellposeExecutor().execute(task,input_path=wrong_signal)
    alternate.write_bytes(weights.read_bytes())
    assert verify_task_inputs(task,Path(task['input_path']),alternate)==task['content_identity']
    bad_shape={**task,'signal':{**task['signal'],'shape':[2,49,37]}}
    with pytest.raises(ValueError,match='shape, axes or dtype'):
        verify_task_inputs(bad_shape,Path(task['input_path']),weights)
    labels=np.ones((37,49),np.uint32)
    complete_fixture_task(first,labels)
    receipt_path=Path(task['result_path'])
    receipt=json.loads(receipt_path.read_text());receipt['content_identity']['input']['sha256']='wrong'
    write_json(receipt_path,receipt)
    with pytest.raises(ValueError,match='different input or weights'):
        run_case(source,record,settings,mapping)


def test_valid_source_mask_does_not_require_weights(tmp_path):
    labels=np.ones((37,49),np.uint32)
    source,record,settings,mapping=setup_case(tmp_path,labels=labels)
    settings=replace(settings,weights=tmp_path/'absent_checkpoint')
    assert 'expression' in run_case(source,record,settings,mapping).products


def test_confirmed_upstream_normalization_is_not_reapplied(tmp_path):
    pixels=np.arange(37*49,dtype=np.float32).reshape(1,37,49)/4000
    labels=np.ones((37,49),np.uint32)
    source,record,settings,mapping=setup_case(tmp_path,pixels,labels)
    source=replace(source,normalized_q995=(4000.0,),intensity_description='confirmed same-domain P0/P99.5')
    result=run_case(source,record,settings,mapping)
    with open_source(tiff_source(Path(result.products['image']),1)) as reader:
        np.testing.assert_array_equal(reader.read_window([0],(0,0,49,37)),pixels)
    assert pd.read_parquet(result.products['channels']).q995.iloc[0]==4000


class FixtureExecutor:
    core_size,halo,match_iou=1024,128,0.5

    def __init__(self):
        self.calls=0

    def execute(self,task):
        from cellweft.data.preparation.segmentation import verify_task_inputs
        verify_task_inputs(task,Path(task['input_path']),Path(task['weights']))
        self.calls+=1
        path=Path(task['result_path'])
        save_tiff(path.parent/'mask.tiff',np.ones(tuple(task['signal']['shape'][1:]),np.uint32))
        write_json(path,{'task_id':task['task_id'],'request':task['request'],
                        'parameters':task['parameters'],'stitching':task['stitching'],
                        'content_identity':task['content_identity'],
                        'mask':'mask.tiff','whole_cell_confirmed':True})
        return path


def test_required_overview_waits_before_any_segmentation(tmp_path):
    labels=np.ones((37,49),np.uint32)
    source,record,settings,mapping=setup_case(tmp_path,labels=labels)
    source=replace(source,mask=replace(source.mask,overview_required=True))
    weights=tmp_path/'weights';weights.write_text('fixture')
    settings=replace(settings,weights=weights)
    executor=FixtureExecutor()
    first=run_case(source,record,settings,mapping,executor)
    assert executor.calls==0 and 'mask' not in first.products
    assert any('overview judgment' in gap for gap in first.gaps)
    assert json.loads(first.metadata_path.read_text())['segmentation_task'] is None
    accepted=replace(source,mask=replace(source.mask,overview_judgment=True))
    second=run_case(accepted,record,settings,mapping,executor)
    assert executor.calls==0 and 'expression' in second.products
    rejected=replace(source,mask=replace(source.mask,overview_judgment=False))
    third=run_case(rejected,record,settings,mapping,executor)
    assert executor.calls==1 and json.loads(third.metadata_path.read_text())['mask_origin']=='cpsam_v2'


def test_missing_task_json_restored_from_metadata(tmp_path):
    source,record,settings,mapping=setup_case(tmp_path)
    weights=tmp_path/'weights';weights.write_text('fixture')
    settings=replace(settings,weights=weights)
    first=run_case(source,record,settings,mapping)
    task=json.loads(first.metadata_path.read_text())['segmentation_task']
    path=Path(task['input_path']).parent/'task.json';path.unlink()
    run_case(source,record,settings,mapping)
    assert json.loads(path.read_text())==task


@pytest.mark.parametrize('remove_signal',[False,True])
def test_incomplete_result_recovers_locally(tmp_path,remove_signal):
    source,record,settings,mapping=setup_case(tmp_path)
    weights=tmp_path/'weights';weights.write_text('fixture')
    settings=replace(settings,weights=weights)
    first=run_case(source,record,settings,mapping)
    task=complete_fixture_task(first,np.ones((37,49),np.uint32))
    (Path(task['result_path']).parent/'mask.tiff').unlink()
    if remove_signal:Path(task['input_path']).unlink()
    executor=FixtureExecutor()
    second=run_case(source,record,settings,mapping,executor)
    assert executor.calls==1 and 'expression' in second.products
    new_task=json.loads(second.metadata_path.read_text())['segmentation_task']
    assert (new_task['task_id']!=task['task_id'])==remove_signal


def test_incomplete_result_remains_pending_for_remote_worker(tmp_path):
    source,record,settings,mapping=setup_case(tmp_path)
    weights=tmp_path/'weights';weights.write_text('fixture')
    settings=replace(settings,weights=weights)
    first=run_case(source,record,settings,mapping)
    task=complete_fixture_task(first,np.ones((37,49),np.uint32))
    (Path(task['result_path']).parent/'mask.tiff').unlink()
    second=run_case(source,record,settings,mapping)
    assert 'mask' not in second.products and 'segmentation result pending' in second.gaps
    executor=FixtureExecutor();executor.execute(task)
    third=run_case(source,record,settings,mapping)
    assert 'expression' in third.products


def test_content_identity_observes_overwrite_with_coarse_timestamps(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from cellweft.data.preparation.segmentation import content_identity
    path=tmp_path/'checkpoint';path.write_bytes(b'checkpoint_a')
    original_stat=Path.stat

    def coarse_stat(current,*args,**kwargs):
        if current==path:
            return SimpleNamespace(st_size=12,st_mtime_ns=0,st_ctime_ns=0)
        return original_stat(current,*args,**kwargs)

    monkeypatch.setattr(Path,'stat',coarse_stat)
    before=content_identity(path)
    path.write_bytes(b'checkpoint_b')
    assert content_identity(path)!=before


@pytest.mark.parametrize('missing_expression',[False,True])
def test_prior_pending_overview_prediction_is_not_reused(tmp_path,missing_expression):
    labels=np.ones((37,49),np.uint32)
    source,record,settings,mapping=setup_case(tmp_path,labels=labels)
    source=replace(source,mask=replace(source.mask,overview_required=True))
    weights=tmp_path/'weights';weights.write_text('fixture')
    settings=replace(settings,weights=weights)
    executor=FixtureExecutor()
    waiting=run_case(source,record,settings,mapping,executor)
    pending_metadata=json.loads(waiting.metadata_path.read_text())
    rejected=replace(source,mask=replace(source.mask,overview_judgment=False))
    completed=run_case(rejected,record,settings,mapping,executor)
    legacy=json.loads(completed.metadata_path.read_text())
    # Reproduce metadata from the previous bug: prediction published under the
    # still-unresolved source overview request, including its original check.
    for key in ('request','mask_request','source_mask_check'):
        legacy[key]=pending_metadata[key]
    legacy['segmentation_task']['request']=legacy['request']
    write_json(completed.metadata_path,legacy)
    if missing_expression:Path(completed.products['expression']).unlink()
    recovered=run_case(source,record,settings,mapping,executor)
    assert executor.calls==1 and 'mask' not in recovered.products
    assert {'image','channels'}<=recovered.products.keys()
    assert any('overview judgment' in gap for gap in recovered.gaps)
    assert json.loads(recovered.metadata_path.read_text())['segmentation_task'] is None
