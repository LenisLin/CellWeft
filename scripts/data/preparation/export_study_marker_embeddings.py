"""Export one study marker-vector asset from a current reviewed D01 scope."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Recheck the selected scope, build CPU vectors, and publish through the shared gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-id", action="append")
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--nas-lock-root", type=Path,
                        help="shared NAS lock directory used by image and vector publishers")
    args = parser.parse_args()

    import pandas as pd
    import torch
    torch.set_num_threads(4)
    from cellweft.data.preparation.embeddings import export_study_embedding_task
    from cellweft.data.io.publication import resource_lock
    from cellweft.data.preparation.io_coordination import default_io_root

    ids = args.image_id
    if ids is None:
        ids = pd.read_parquet(args.study_root / "images.parquet").image_id.astype(str).tolist()
    locks = args.nas_lock_root or default_io_root(args.data_root)
    task = {
        "study": args.study_root.name,
        "image_ids": ids,
        "panel_ids": [],
        "scopes": [{"scope_key": "direct", "image_ids": ids}],
    }
    with resource_lock(locks, name="embedding_controller.lock"):
        result = export_study_embedding_task(
            task,
            data_root=args.data_root,
            s1_path=args.s1,
            bank_dir=args.bank,
            nas_lock_root=locks,
            header_cache={},
        )
    print(args.study_root / "marker_embeddings.pt")
    return result


if __name__ == "__main__":
    main()
