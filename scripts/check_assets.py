#!/usr/bin/env python3
"""Check bundled data and asset integrity without starting Isaac Sim."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        os.environ.get("ADAHVLA_ROOT", Path(__file__).resolve().parents[1])
    ))
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "configs/locomotion.json").read_text())
    paths = {name: root / value for name, value in config["paths"].items()}
    manifest = json.loads((root / "assets/manifest.json").read_text())
    errors = []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file():
            errors.append(f"Missing: {name}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != expected["size"] or digest.hexdigest() != expected["sha256"]:
            errors.append(f"Content changed: {name}")

    for name, path in paths.items():
        if not path.exists():
            errors.append(f"Configured {name} path does not exist: {path}")
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")

    with gzip.open(paths["dataset"], "rt") as stream:
        episodes = json.load(stream)["episodes"]
    scenes = sorted({Path(episode["scene_id"]).stem for episode in episodes})
    for scene in scenes:
        path = paths["scenes"] / scene / f"{scene}.usd"
        if not path.is_file():
            errors.append(f"Missing benchmark scene: {path}")
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    size = sum(item["size"] for item in manifest["files"].values())
    print(f"OK: {len(episodes)} episodes, {len(scenes)} scenes, "
          f"{len(manifest['files'])} verified files ({size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
