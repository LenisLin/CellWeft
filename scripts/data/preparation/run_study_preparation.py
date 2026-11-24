"""Prepare reviewed per-study input TIFFs through shared source modules.

Inputs: complete study tables, revision-bound D01 acceptance, mask binding JSON.
Outputs: per-image shared pixels, masks, expression, spatial tables and metadata.
Side effects: writes selected image products/cache; never rewrites images.parquet.
Prerequisites: reviewed inputs; explicit mask status and per-panel tissue review.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from cellweft.data.io.readers import open_source
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.configuration import load_settings
from cellweft.data.preparation.inputs import load_reviewed_study_inputs, assert_reviewed_binding
from cellweft.data.preparation.pipeline import prepare_image


NODE_SETTINGS = {'226':('bf16',32),'73':('bf16',16),'72':('fp32',8),'local':('bf16',16)}


def main() -> None:
    """Bind one approved input scope and run/resume CPU preparation plus file handoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--image-id',action='append')
    parser.add_argument('--mask-bindings',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--settings',type=Path,default=Path(__file__).resolve().parents[3]/'configs/data/preparation/defaults.yaml')
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--node',choices=NODE_SETTINGS,default='local')
    parser.add_argument('--tissue-reviewed-panel',action='append',default=[])
    args = parser.parse_args()
    inputs = load_reviewed_study_inputs(args.study_root,data_root=args.data_root,image_ids=args.image_id)
    bindings = json.loads(args.mask_bindings.read_text())
    precision,batch = NODE_SETTINGS[args.node]
    settings = load_settings(args.settings,args.cache_dir,weights=args.weights,batch_size=batch,precision=precision)
    for row in inputs.images.itertuples():
        if row.image_id not in bindings:
            raise ValueError(f'{row.image_id}: explicit unknown/absent/bound mask status required')
        # The study read lease protects input/panel revision through image publication.
        with resource_lock(args.study_root,shared=True), resource_lock((args.data_root/row.input_image_path).parent,shared=True,name=".input.lock"):
            assert_reviewed_binding(args.study_root,data_root=args.data_root,binding=inputs.binding)
            record,source,panel = inputs.source(row.image_id,bindings[row.image_id])
            with open_source(source) as reader:
                result = prepare_image(record,source,reader,panel,
                                       replace(settings,sampling_reviewed=row.panel_id in args.tissue_reviewed_panel))
            print(row.image_id,result.metadata_path,', '.join(result.gaps))


if __name__ == '__main__':
    main()
