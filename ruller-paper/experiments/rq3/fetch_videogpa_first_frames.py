#!/usr/bin/env python3
"""Fetch only the first frame for the official VideoGPA DL3DV protocol.

DL3DV stores each scene as a ZIP. RemoteZip uses HTTP range requests, so this
does not materialize the roughly 50--125 MB archive for every selected scene.
The output layout matches VideoGPA's ``replicate.py`` expectations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

from huggingface_hub import get_token
from remotezip import RemoteZip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = json.loads(args.prompts.read_text())
    selected = list(prompts)[: args.count]
    token = get_token()
    if not token:
        raise RuntimeError("No Hugging Face token is configured")

    headers = {"Authorization": f"Bearer {token}"}
    manifest = []
    for index, key in enumerate(selected, 1):
        parts = PurePosixPath(key).parts
        if len(parts) != 3:
            raise ValueError(f"Unexpected VideoGPA key: {key}")
        split, scene_hash, image_level = parts
        destination = args.output / key / "frame_00001.png"
        if destination.exists():
            print(f"[{index}/{len(selected)}] exists {scene_hash}", flush=True)
            manifest.append({"key": key, "frame": str(destination)})
            continue

        url = (
            "https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P/"
            f"resolve/main/{split}/{scene_hash}.zip"
        )
        with RemoteZip(url, headers=headers) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if f"/{image_level}/" in f"/{name}"
                and PurePosixPath(name).stem == "frame_00001"
            ]
            if not candidates:
                raise FileNotFoundError(f"frame_00001 under {image_level}: {scene_hash}")
            payload = archive.read(candidates[0])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        print(f"[{index}/{len(selected)}] fetched {scene_hash}", flush=True)
        manifest.append({"key": key, "frame": str(destination), "zip_member": candidates[0]})

    (args.output / "first100_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
