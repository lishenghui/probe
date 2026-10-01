#!/usr/bin/env python3
"""Evaluate video quality metrics across VideoGPA-generated video sets."""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import lpips
except ImportError:
    lpips = None

try:
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    from skimage.metrics import structural_similarity as compute_ssim
except ImportError:
    compute_psnr = None
    compute_ssim = None


def read_video_frames(video_path: Path, max_frames=49):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def evaluate_video(frames, lpips_model=None, device="cpu"):
    if not frames:
        return {}

    num_frames = len(frames)
    first_frame = frames[0]

    psnr_first = []
    ssim_first = []
    psnr_consec = []
    ssim_consec = []

    # Frame-to-frame & First-frame metrics
    for i in range(1, num_frames):
        cur = frames[i]
        prev = frames[i - 1]

        if compute_psnr is not None:
            psnr_first.append(compute_psnr(first_frame, cur, data_range=255))
            psnr_consec.append(compute_psnr(prev, cur, data_range=255))
        if compute_ssim is not None:
            ssim_first.append(compute_ssim(first_frame, cur, channel_axis=2, data_range=255))
            ssim_consec.append(compute_ssim(prev, cur, channel_axis=2, data_range=255))

    # LPIPS perceptual temporal coherence
    lpips_consec = []
    if lpips_model is not None and num_frames > 1:
        tensors = [
            torch.from_numpy(f).permute(2, 0, 1).float() / 127.5 - 1.0
            for f in frames
        ]
        batch_prev = torch.stack(tensors[:-1]).to(device)
        batch_cur = torch.stack(tensors[1:]).to(device)
        with torch.no_grad():
            dists = lpips_model(batch_prev, batch_cur).squeeze().cpu().numpy()
            if dists.ndim == 0:
                lpips_consec = [float(dists)]
            else:
                lpips_consec = dists.tolist()

    # Motion magnitude (flow norm approximation via frame diff)
    motion_diffs = [
        np.mean(np.abs(frames[i].astype(float) - frames[i - 1].astype(float)))
        for i in range(1, num_frames)
    ]

    return {
        "num_frames": num_frames,
        "psnr_consec_mean": float(np.mean(psnr_consec)) if psnr_consec else None,
        "ssim_consec_mean": float(np.mean(ssim_consec)) if ssim_consec else None,
        "psnr_first_mean": float(np.mean(psnr_first)) if psnr_first else None,
        "ssim_first_mean": float(np.mean(ssim_first)) if ssim_first else None,
        "lpips_temporal_mean": float(np.mean(lpips_consec)) if lpips_consec else None,
        "motion_energy": float(np.mean(motion_diffs)) if motion_diffs else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    v_dir = args.results_root / args.variant
    video_paths = sorted(v_dir.glob("*/seed_456_dpo_w1.0.mp4"))
    if not video_paths:
        video_paths = sorted(v_dir.glob("*/*.mp4"))

    print(f"Found {len(video_paths)} videos in {v_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = None
    if lpips is not None:
        try:
            lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
            print("Loaded LPIPS model on", device)
        except Exception as e:
            print("Failed to load LPIPS:", e)

    results = {}
    for vp in video_paths:
        scene_hash = vp.parent.name
        frames = read_video_frames(vp)
        metrics = evaluate_video(frames, lpips_model=lpips_model, device=device)
        results[scene_hash] = metrics

    # Compute aggregate summaries
    summary = {
        "variant": args.variant,
        "total_videos": len(results),
        "mean_psnr_consec": float(np.nanmean([m["psnr_consec_mean"] for m in results.values() if m.get("psnr_consec_mean") is not None])) if results else None,
        "mean_ssim_consec": float(np.nanmean([m["ssim_consec_mean"] for m in results.values() if m.get("ssim_consec_mean") is not None])) if results else None,
        "mean_psnr_first": float(np.nanmean([m["psnr_first_mean"] for m in results.values() if m.get("psnr_first_mean") is not None])) if results else None,
        "mean_ssim_first": float(np.nanmean([m["ssim_first_mean"] for m in results.values() if m.get("ssim_first_mean") is not None])) if results else None,
        "mean_lpips_temporal": float(np.nanmean([m["lpips_temporal_mean"] for m in results.values() if m.get("lpips_temporal_mean") is not None])) if results else None,
        "mean_motion_energy": float(np.nanmean([m["motion_energy"] for m in results.values() if m.get("motion_energy") is not None])) if results else None,
    }

    output_payload = {
        "summary": summary,
        "items": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output_payload, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
