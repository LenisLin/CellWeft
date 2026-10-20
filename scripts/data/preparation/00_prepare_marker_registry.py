"""Apply the manually confirmed global marker identities.

Inputs: confirmed CSV with marker_id, display_name, canonical_protein_id, measured_form.
Outputs: NAS metadata/marker_registry.csv.
Prerequisite: source markers have been reviewed in one common semantic space.
"""
import argparse
from pathlib import Path

import pandas as pd

from cellweft.data.preparation.markers import build_marker_registry

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--confirmed", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

# Preserve confirmed measurement identities, including families and combinations.
registry = build_marker_registry(pd.read_csv(args.confirmed))

# Publish the semantic input used by every dataset entrypoint.
args.output.parent.mkdir(parents=True, exist_ok=True)
registry.to_csv(args.output, index=False)
