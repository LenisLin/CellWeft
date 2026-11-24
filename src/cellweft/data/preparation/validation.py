"""Reopen persisted preparation products and validate their direct contracts."""
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def validate_products(products: dict[str, str], shape_yx: tuple[int, int]) -> None:
    """Validate TIFF/channel and cell/expression/neighbor keys before publication/reuse."""
    tables = {key:pd.read_parquet(path) for key,path in products.items() if key in {'channels','cells','expression','neighbors'}}
    for key in ('image','mask'):
        if key in products:
            with tifffile.TiffFile(products[key]) as tif:
                dtype = np.dtype('float32' if key == 'image' else 'uint32')
                count = int(tables['channels'].tiff_page.notna().sum()) if key == 'image' else 1
                if len(tif.pages) != count or any(page.shape != tuple(shape_yx) or page.dtype != dtype for page in tif.pages):
                    raise ValueError(f"{key}: persisted TIFF shape/dtype/pages violate contract")
    if 'channels' in tables:
        channels = tables['channels']
        active = channels[channels.tiff_page.notna()]
        if sorted(active.tiff_page.astype(int)) != list(range(len(active))) or not (active.q995 > 0).all():
            raise ValueError("Channel pages or shared normalization denominators are invalid")
        if active.loc[active.role != 'downstream','expression_column'].notna().any():
            raise ValueError("Segmentation-only channel has an expression reference")
    if 'cells' in tables:
        cells = tables['cells']
        if cells.cell_index.tolist() != list(range(len(cells))) or cells.mask_label.tolist() != sorted(set(cells.mask_label)) or (cells.mask_label <= 0).any():
            raise ValueError("Cell index/label contract is invalid")
        if (cells.area <= 0).any():
            raise ValueError("Cells must contain observed mask pixels")
        from cellweft.data.io.readers import open_source, tiff_source
        from cellweft.data.preparation.crops import grid_origins, scan_mask_geometry
        if 'mask' not in products: raise ValueError("Cells require a final mask")
        with open_source(tiff_source(Path(products['mask']),1)) as reader:
            counts, expected_origins = scan_mask_geometry(reader, block_size=512)
        if counts != dict(zip(cells.mask_label.astype(int),cells.area.astype(int))):
            raise ValueError("Cell labels/areas differ from final mask")
        grid = set(grid_origins(tuple(shape_yx)))
        for label, origins in zip(cells.mask_label.astype(int), cells.crop_origins):
            actual = [(int(origin['x0']), int(origin['y0'])) for origin in origins]
            if any(origin not in grid for origin in actual):
                raise ValueError("Cell crop origin lies outside the complete native grid")
            if len(actual) != len(set(actual)) or set(actual) != expected_origins.get(label, set()):
                raise ValueError("Cell crop origins do not match positive mask pixels")
    for key in ('expression','neighbors'):
        if key in tables:
            if 'cells' not in tables: raise ValueError(f"{key} requires cell table")
            table = tables[key]
            if table.cell_index.duplicated().any() or set(table.cell_index) != set(cells.cell_index):
                raise ValueError(f"{key}: cell keys differ from cell table")
            if key == 'expression':
                joined = table.set_index('cell_index').loc[cells.cell_index]
                if joined.mask_label.tolist() != cells.mask_label.tolist():
                    raise ValueError("Expression labels do not match cell keys")
                expected = tables['channels'].expression_column.dropna().tolist()
                if set(table.columns)-{'cell_index','mask_label'} != set(expected):
                    raise ValueError("Expression columns differ from selected downstream markers")
                if not np.isfinite(table[expected].to_numpy()).all():
                    raise ValueError("Expression contains nonfinite values")
            else:
                columns = [f'n{i:02d}' for i in range(1,21)]
                values = table[columns].to_numpy()
                if not np.isin(values,[-1,*cells.cell_index]).all() or np.any(values == table.cell_index.to_numpy()[:,None]):
                    raise ValueError("Neighbor index is invalid or contains self")
