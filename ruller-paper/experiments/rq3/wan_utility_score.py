#!/usr/bin/env python3
"""Cheap, reproducible first-pass metrics for generated Wan utility clips.

These metrics are headroom gates, not replacements for the official VBench or
PanoWan evaluation. The script scores saved clips so generation never needs to
be repeated when a stronger evaluator is added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_frames(path: Path, max_frames: int) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in indices]
    return frames


def cheap_metrics(frames: list[np.ndarray]) -> dict[str, float]:
    import cv2

    values = [frame.astype(np.float32) / 255.0 for frame in frames]
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in values]
    sharpness = float(np.mean([cv2.Laplacian(g, cv2.CV_32F).var() for g in gray]))

    # An equirectangular panorama is periodic horizontally. Compare narrow edge
    # strips after reversing neither side: corresponding boundary columns meet.
    seam = []
    for frame in values:
        strip = max(2, frame.shape[1] // 128)
        seam.append(float(np.abs(frame[:, :strip] - frame[:, -strip:]).mean()))
    seam_error = float(np.mean(seam))

    frame_delta = []
    flow_magnitude = []
    for previous, current in zip(gray, gray[1:]):
        frame_delta.append(float(np.abs(current - previous).mean()))
        flow = cv2.calcOpticalFlowFarneback(
            (previous * 255).astype(np.uint8), (current * 255).astype(np.uint8),
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        flow_magnitude.append(float(np.linalg.norm(flow, axis=-1).mean()))
    temporal_delta = float(np.mean(frame_delta)) if frame_delta else 0.0
    dynamic_degree = float(np.mean(flow_magnitude)) if flow_magnitude else 0.0
    return {
        "panorama_seam": -seam_error,
        "panorama_seam_error": seam_error,
        "imaging_quality": sharpness,
        "sharpness_laplacian": sharpness,
        "temporal_consistency_proxy": -temporal_delta,
        "temporal_frame_delta": temporal_delta,
        "dynamic_degree": dynamic_degree,
        "slow_motion_control": -dynamic_degree,
    }


_CLIP_CACHE = None


def clip_metrics(frames: list[np.ndarray], prompt: str, model_path: Path) -> dict[str, float]:
    global _CLIP_CACHE
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if _CLIP_CACHE is None:
        model = CLIPModel.from_pretrained(model_path).to("cuda").eval()
        processor = CLIPProcessor.from_pretrained(model_path)
        _CLIP_CACHE = model, processor
    model, processor = _CLIP_CACHE
    images = [Image.fromarray(frame) for frame in frames]
    texts = [prompt, "a beautiful, aesthetically pleasing, high quality cinematic video",
             "an ugly, low quality, poorly composed video"]
    inputs = processor(text=texts, images=images, padding=True, truncation=True,
                       return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model(**inputs)
    image = torch.nn.functional.normalize(output.image_embeds, dim=-1)
    text = torch.nn.functional.normalize(output.text_embeds, dim=-1)
    similarity = image @ text.T
    return {
        "text_alignment_clip": float(similarity[:, 0].mean()),
        "aesthetic_quality": float((similarity[:, 1] - similarity[:, 2]).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--clip", type=Path)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    pattern = f"{args.adapter}/*/*.json" if args.adapter else "*/*/*.json"
    records = []
    for sidecar in sorted(args.results_root.glob(pattern)):
        record = json.loads(sidecar.read_text())
        video = Path(record["video"])
        if not video.is_absolute() and not video.exists():
            video = sidecar.parent / video
        frames = load_frames(video, args.max_frames)
        metrics = cheap_metrics(frames)
        if args.clip is not None:
            metrics.update(clip_metrics(frames, record["prompt"], args.clip))
        primary = record["primary_metric"]
        record["metrics"] = metrics
        record["primary_value"] = metrics.get(primary)
        record["primary_available"] = primary in metrics
        records.append(record)
        print(f"{record['adapter']} {record['variant']} p{record['prompt_index']} "
              f"primary={record['primary_value']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {args.output} ({len(records)} clips)")


if __name__ == "__main__":
    main()
