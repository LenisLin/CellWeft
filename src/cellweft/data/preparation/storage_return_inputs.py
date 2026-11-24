"""Expose accepted NAS input assets in a returned image bundle without rewriting them."""
from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.inputs import file_revision
from cellweft.data.preparation.io_coordination import formal_write_lease


def retain_image_inputs(study_root: Path, image_id: str, image_record: dict,
                        accepted: dict, *, data_root: Path, lock_root: Path,
                        prepublish_check: Callable[[], None]) -> list[dict]:
    """Hard-link a nonstandard existing NAS input; never replace an existing asset."""
    if Path(image_id).name != image_id or image_id in {'.', '..'}:
        raise ValueError('Image ID must be a single path component')
    target = study_root / 'images' / image_id
    identities = {value['path']: value for value in accepted['source_input_identity']}
    assets = []
    for field, filename in [('input_image_path', 'input.tiff'),
                            ('valid_pixels_path', 'valid_pixels.tiff'),
                            ('sample_region_path', 'sample_region.tiff')]:
        value = image_record.get(field)
        if not isinstance(value, str) or not value:
            if field == 'input_image_path':
                raise ValueError(f'{image_id}: authoritative input_image_path is missing')
            continue
        source = Path(value)
        source = source if source.is_absolute() else data_root / source
        source = source.resolve()
        if not source.is_relative_to(data_root.resolve()):
            raise ValueError(f'{image_id}: input reference is outside the protected NAS')
        expected = identities.get(str(source))
        if expected is None:
            raise ValueError(f'{image_id}: input/conditional asset lacks accepted version evidence')
        assets.append((source, target / filename, expected))
    if all(source == destination.resolve() and file_revision(source) == expected
           for source, destination, expected in assets):
        return [{'path': str(destination.resolve()), 'source': expected, 'new_hardlink': False}
                for source, destination, expected in assets]
    result = []
    with formal_write_lease(study_root, io_lock_root=lock_root):
        prepublish_check()
        with resource_lock(target, name='.input.lock'):
            with resource_lock(target, name='.preparation.lock'):
                with resource_lock(target):
                    prepublish_check()
                    for source, destination, expected in assets:
                        if file_revision(source) != expected:
                            raise ValueError(f'{image_id}: accepted NAS input identity changed')
                        linked = False
                        if destination.exists():
                            if not os.path.samefile(source, destination):
                                raise ValueError(f'{image_id}: existing canonical input conflicts; retained unchanged')
                        else:
                            os.link(source, destination)
                            linked = True
                            if file_revision(source) != expected or not os.path.samefile(source, destination):
                                destination.unlink()
                                raise ValueError(f'{image_id}: input changed during hard-link publication')
                        result.append({'path': str(destination.resolve()), 'source': expected,
                                       'new_hardlink': linked})
                    descriptor = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
    return result
