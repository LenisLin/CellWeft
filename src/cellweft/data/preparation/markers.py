"""Apply manually confirmed identities and source-channel roles."""
from __future__ import annotations

import pandas as pd


REGISTRY_COLUMNS = ["marker_id", "display_name", "canonical_protein_id", "measured_form"]
PANEL_COLUMNS = ["panel_id", "source_channel_index", "source_name", "marker_id",
                 "role", "nuclear", "body", "exclusion_reason"]


def build_marker_registry(confirmed: pd.DataFrame) -> pd.DataFrame:
    """Validate the manually confirmed semantic registry; family IDs may have no protein ID."""
    result = confirmed[REGISTRY_COLUMNS].copy()
    if result.marker_id.isna().any() or result.marker_id.duplicated().any():
        raise ValueError("Registry requires unique confirmed marker_id values")
    return result


def build_panel_mapping(confirmed: pd.DataFrame, registry: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Resolve source-channel roles against the registry, retaining excluded channels.

    Output page indices are determined per image after numerical filtering.
    """
    mapping = confirmed[PANEL_COLUMNS].copy()
    for key in ("nuclear", "body"):
        converted = mapping[key].astype(str).str.lower().map({"true": True, "false": False})
        if converted.isna().any():
            raise ValueError(f"{key} must contain explicit true/false values")
        mapping[key] = converted
    if not mapping.role.isin(["downstream", "segmentation-only", "exclude"]).all():
        raise ValueError("Unknown channel role")
    active = mapping.role != "exclude"
    if not mapping.loc[active, "marker_id"].isin(registry.marker_id).all():
        raise ValueError("Every retained measurement must reference the confirmed registry")
    if mapping.loc[~active, ["nuclear", "body"]].to_numpy().any():
        raise ValueError("Excluded channels cannot be segmentation group members")
    output = {}
    for panel, rows in mapping.groupby("panel_id", sort=False):
        rows = rows.sort_values("source_channel_index").reset_index(drop=True)
        indices = rows.source_channel_index.to_numpy()
        if list(indices) != list(range(len(rows))):
            raise ValueError(f"{panel}: map all source channels once in their original order")
        proteins = rows[rows.role == "downstream"].marker_id
        if proteins.duplicated().any():
            raise ValueError(f"{panel}: duplicate protein measurements need manual channel exclusion")
        output[str(panel)] = rows
    return output
