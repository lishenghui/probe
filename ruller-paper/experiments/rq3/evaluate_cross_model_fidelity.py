#!/usr/bin/env python3
"""Compute cross-model output fidelity (variant vs original) and text-video prompt adherence."""

import argparse
import json
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

try:
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
except ImportError:
    CLIPModel = None


def read_video_frames(video_path: Path, max_frames=49):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--captions-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load LPIPS
    lpips_net = None
    if lpips is not None:
        try:
            lpips_net = lpips.LPIPS(net="vgg").to(device).eval()
            print("LPIPS VGG network loaded successfully.")
        except Exception as e:
            print("LPIPS loading failed:", e)

    # Load CLIP
    clip_model = None
    clip_processor = None
    if CLIPModel is not None:
        try:
            clip_name = "openai/clip-vit-base-patch32"
            clip_model = CLIPModel.from_pretrained(clip_name).to(device).eval()
            clip_processor = CLIPProcessor.from_pretrained(clip_name)
            print("CLIP model loaded successfully.")
        except Exception as e:
            print("CLIP loading failed:", e)

    # Load captions
    captions = json.loads(args.captions_json.read_text())
    caption_by_hash = {}
    for key, text in captions.items():
        parts = key.split("/")
        if len(parts) >= 2:
            caption_by_hash[parts[1]] = text.strip()

    orig_dir = args.results_root / "original"
    orig_videos = {p.parent.name: p for p in orig_dir.glob("*/seed_456_dpo_w1.0.mp4")}
    print(f"Found {len(orig_videos)} original reference videos.")

    all_variants_data = {}
    variants = ["original", "e99", "e95", "e90", "base"]

    for variant in variants:
        v_dir = args.results_root / variant
        v_videos = {p.parent.name: p for p in v_dir.glob("*/seed_456_dpo_w1.0.mp4")}
        print(f"\n================ Processing variant: {variant} ({len(v_videos)} videos) ================")

        variant_items = {}
        for scene_hash, v_path in v_videos.items():
            frames_var = read_video_frames(v_path)
            if not frames_var:
                continue

            num_frames = len(frames_var)
            item_stats = {"num_frames": num_frames}

            # 1. Prompt adherence (CLIP similarity)
            prompt_text = caption_by_hash.get(scene_hash, "")
            if clip_model is not None and clip_processor is not None and prompt_text:
                try:
                    inputs = clip_processor(
                        text=[prompt_text],
                        images=[frames_var[0], frames_var[num_frames // 2], frames_var[-1]],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)
                    with torch.no_grad():
                        image_embeds = clip_model.get_image_features(inputs["pixel_values"])
                        text_embeds = clip_model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
                        image_embeds = F.normalize(image_embeds, dim=-1)
                        text_embeds = F.normalize(text_embeds, dim=-1)
                        sim = (image_embeds @ text_embeds.T).squeeze(-1).mean().item()
                        item_stats["clip_score"] = float(sim)
                except Exception as e:
                    item_stats["clip_score"] = None
            else:
                item_stats["clip_score"] = None

            # 2. Fidelity vs Original (if original video available)
            if scene_hash in orig_videos:
                frames_orig = read_video_frames(orig_videos[scene_hash])
                min_len = min(len(frames_var), len(frames_orig))

                if min_len > 0:
                    # PSNR & SSIM vs Original
                    psnr_to_orig = []
                    ssim_to_orig = []
                    for t in range(min_len):
                        f_var = frames_var[t]
                        f_orig = frames_orig[t]
                        if compute_psnr is not None:
                            psnr_to_orig.append(compute_psnr(f_orig, f_var, data_range=255))
                        if compute_ssim is not None:
                            ssim_to_orig.append(compute_ssim(f_orig, f_var, channel_axis=2, data_range=255))

                    item_stats["psnr_vs_original"] = float(np.mean(psnr_to_orig)) if psnr_to_orig else None
                    item_stats["ssim_vs_original"] = float(np.mean(ssim_to_orig)) if ssim_to_orig else None

                    # LPIPS vs Original (per-frame perceptual distance)
                    if lpips_net is not None:
                        t_var = torch.stack([
                            torch.from_numpy(frames_var[t]).permute(2, 0, 1).float() / 127.5 - 1.0
                            for t in range(min_len)
                        ]).to(device)
                        t_orig = torch.stack([
                            torch.from_numpy(frames_orig[t]).permute(2, 0, 1).float() / 127.5 - 1.0
                            for t in range(min_len)
                        ]).to(device)
                        with torch.no_grad():
                            dists = lpips_net(t_orig, t_var).squeeze().cpu().numpy()
                            item_stats["lpips_vs_original"] = float(np.mean(dists))

            variant_items[scene_hash] = item_stats

        # Compute summary
        n_items = len(variant_items)
        clip_scores = [it["clip_score"] for it in variant_items.values() if it.get("clip_score") is not None]
        lpips_vs_orig = [it["lpips_vs_original"] for it in variant_items.values() if it.get("lpips_vs_original") is not None]
        ssim_vs_orig = [it["ssim_vs_original"] for it in variant_items.values() if it.get("ssim_vs_original") is not None]
        psnr_vs_orig = [it["psnr_vs_original"] for it in variant_items.values() if it.get("psnr_vs_original") is not None]

        summary = {
            "variant": variant,
            "total_evaluated": n_items,
            "mean_clip_score": float(np.mean(clip_scores)) if clip_scores else None,
            "mean_lpips_vs_original": float(np.mean(lpips_vs_orig)) if lpips_vs_orig else None,
            "mean_ssim_vs_original": float(np.mean(ssim_vs_orig)) if ssim_vs_orig else None,
            "mean_psnr_vs_original": float(np.mean(psnr_vs_orig)) if psnr_vs_orig else None,
        }
        all_variants_data[variant] = {
            "summary": summary,
            "items": variant_items,
        }
        print(f"Summary for {variant}: {json.dumps(summary, indent=2)}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(all_variants_data, indent=2))
    print(f"\nFinal cross-model fidelity report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
