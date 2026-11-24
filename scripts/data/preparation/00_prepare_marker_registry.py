"""Derive shared downstream marker identities from the maintained S1 table.

Inputs: final S1 CSV; no separate manually maintained marker confirmation table.
Outputs: the marker_registry.csv member of NAS raw/shared/metadata.json.
Side effects: atomically replaces the shared metadata bundle.
Prerequisites: S1 measurement identities have been reviewed.
"""
import argparse
from pathlib import Path
import tempfile
from cellweft.data.preparation.markers import registry_from_s1
from cellweft.data.preparation.io_coordination import formal_write_lease
from cellweft.data.source_metadata import build_metadata_bundle, metadata_source_directory, publish_metadata_sources


def main() -> None:
    """Resolve S1 identities through shared code and save the requested registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--s1',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    registry = registry_from_s1(args.s1)
    if args.output.name != 'metadata.json' or args.output.parent.name != 'shared':
        raise ValueError('--output must be raw/shared/metadata.json')
    if args.output.exists():
        directory = metadata_source_directory(args.output.parents[2], '_shared', scope='shared')
        registry.to_csv(directory / 'marker_registry.csv', index=False, encoding='utf-8-sig')
        publish_metadata_sources(directory)
    else:
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / 'marker_registry.csv'
            registry.to_csv(source, index=False, encoding='utf-8-sig')
            with formal_write_lease(args.output, data_root=args.output.parents[2]):
                build_metadata_bundle(args.output, {'shared/marker_registry.csv': source},
                                      owner={'scope': 'shared', 'study': None})


if __name__ == '__main__':
    main()
