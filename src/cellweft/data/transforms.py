"""One pixel-center geometry for image, instances, measured pixels and cell lookup."""
from __future__ import annotations

import numpy as np
from cellweft.data.contracts import CropSample, ViewGeometry, ViewSample


def apply_view_geometry(sample: CropSample, geometry: ViewGeometry, *,
                        mask_identity: str | None, label_to_cell: dict[int, int]) -> ViewSample:
    """Transform a crop and map visible labels to the same final-mask cell indices.

    Affine matrices map integer pixel centers in input coordinates to view centers.
    Floating interpolation is only for intensities; float64 nearest-exact preserves
    every uint32 label. A mismatched mask identity or unmapped label is an error.
    """
    import torch
    import torch.nn.functional as F
    if sample.mask_identity != mask_identity:
        raise ValueError("View cell lookup belongs to a different final mask")
    if sample.labels is not None and not mask_identity:
        raise ValueError("Labeled views require an explicit final mask identity")
    x0,y0,x1,y1 = geometry.window
    h,w = sample.pixels.shape[-2:]
    n = geometry.size
    if n not in (224,98) or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("View requires a valid crop window and size 224 or 98")

    def resize(values, mode):
        tensor = torch.from_numpy(np.ascontiguousarray(values))[None]
        kwargs = {'align_corners':False,'antialias':True} if mode == 'bicubic' else {}
        result = F.interpolate(tensor,size=(n,n),mode=mode,**kwargs)[0].numpy()
        if geometry.flip_x: result = result[...,::-1]
        if geometry.flip_y: result = result[...,::-1,:]
        return np.rot90(result,geometry.quarter_turns % 4,axes=(-2,-1)).copy()

    pixels = np.clip(resize(sample.pixels[:,y0:y1,x0:x1],'bicubic'),0,1)
    valid = resize(sample.spatial_valid[:,y0:y1,x0:x1].astype(np.float64),'nearest-exact').astype(bool)
    labels = None
    cells = np.empty(0,np.int64)
    if sample.labels is not None:
        labels = resize(sample.labels[None,y0:y1,x0:x1].astype(np.float64),'nearest-exact')[0].astype(np.uint32)
        ids = np.unique(labels); ids = ids[ids > 0]
        if any(int(label) not in label_to_cell for label in ids):
            raise ValueError("Visible label is absent from the final-mask cell table")
        cells = np.array([label_to_cell[int(label)] for label in ids],np.int64)
    sx,sy = n/(x1-x0),n/(y1-y0)
    ox,oy = sample.origin_xy
    affine = np.array([[sx,0,sx*(.5-x0-ox)-.5],[0,sy,sy*(.5-y0-oy)-.5],[0,0,1.]])
    if geometry.flip_x: affine = np.array([[-1,0,n-1],[0,1,0],[0,0,1]]) @ affine
    if geometry.flip_y: affine = np.array([[1,0,0],[0,-1,n-1],[0,0,1]]) @ affine
    for _ in range(geometry.quarter_turns % 4):
        affine = np.array([[0,1,0],[-1,0,n-1],[0,0,1]]) @ affine
    output = CropSample(pixels,labels,valid,sample.origin_xy,mask_identity)
    return ViewSample(output,affine,np.linalg.inv(affine),cells)
