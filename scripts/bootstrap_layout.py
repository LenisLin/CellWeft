"""Create the declared SSD/NAS directories without adding placeholder files."""

import argparse
import json
from pathlib import Path


def create_layout(project_root: Path, data_root: Path | None) -> int:
    """Create missing directories and return their count; existing files are untouched."""
    layout_path = Path(__file__).resolve().parents[1] / "configs" / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    roots = {"ssd": project_root}
    if data_root is not None:
        roots["nas"] = data_root
    created = 0
    for group, root in roots.items():
        for relative_path in layout[group]:
            directory = root / relative_path
            existed = directory.is_dir()
            directory.mkdir(parents=True, exist_ok=True)
            created += not existed
    return created


def main() -> None:
    """Parse explicit destinations and restore the versioned directory layout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1],
        help="SSD project root; defaults to this checkout",
    )
    parser.add_argument(
        "--data-root", type=Path,
        help="NAS root; omitted means create only the SSD layout",
    )
    args = parser.parse_args()
    created = create_layout(args.project_root, args.data_root)
    print(f"Created {created} directories.")


if __name__ == "__main__":
    main()
