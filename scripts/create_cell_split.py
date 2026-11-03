"""Create the tracked public-safe CELL student split artifact."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell import build_split_artifact, load_cell_manifest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    records = load_cell_manifest(arguments.manifest)
    artifact = build_split_artifact(records)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
