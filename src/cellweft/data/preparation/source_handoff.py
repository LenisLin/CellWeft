"""Validate a source owner before consuming its immutable completed ranges."""
from typing import Any, Literal, Mapping


def source_owner_mode(
    observation: Mapping[str, Any], expected_owner: Mapping[str, Any]
) -> Literal["live", "complete", "retained_partial"]:
    """Require the same live owner or a verified terminal with no remaining writer.

    A partial terminal permits checking its complete ranges; consumers must still
    prove full coverage, current source versions and CRC for every selected core.
    """
    if observation["worker"] != expected_owner:
        raise ValueError("Source owner differs from the registered owner")
    if observation["live"]:
        if not observation["command_matches"] or not observation["private_lock_exists"]:
            raise ValueError("Live source owner lacks its command identity or private lock")
        return "live"
    terminal = observation["terminal"]
    if (observation["private_lock_exists"] or observation["process_group_members"]
            or terminal is None or not terminal["source_metadata_reverified"]):
        raise ValueError("Closed source owner requires an empty group, removed lock and verified terminal")
    if (terminal["identity"] != observation["progress"]["identity"]
            or terminal["identity"]["process_group"] != expected_owner["process_group"]):
        raise ValueError("Source terminal identity differs from the original process group")
    tasks = terminal["tasks"]
    if (terminal["requested_bytes"] != sum(row["bytes"] for row in tasks)
            or terminal["received_bytes"] != sum(row["received"] for row in tasks)
            or any(not 0 <= row["received"] <= row["bytes"] for row in tasks)):
        raise ValueError("Source terminal byte counts are inconsistent")
    complete = all(row["status"] == "complete" and row["received"] == row["bytes"] for row in tasks)
    mode: Literal["complete", "retained_partial"] = "complete" if complete else "retained_partial"
    if terminal["status"] != mode:
        raise ValueError("Source terminal status differs from its actual range coverage")
    return mode
