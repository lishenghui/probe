#!/bin/bash
set -uo pipefail
cd /nobackup/proj/disk/bloom/personal/shenghui/data/models/Wan2.1-I2V-14B-480P-comfy-bf16
for f in split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors split_files/text_encoders/umt5_xxl_fp16.safetensors split_files/clip_vision/clip_vision_h.safetensors split_files/vae/wan_2.1_vae.safetensors; do
  sha=$(/usr/bin/python3 -c "import json;print([s['lfs']['sha256'] for s in json.load(open('hub_listing.json'))['siblings'] if s['rfilename']=='$f'][0])")
  mkdir -p "$(dirname "$f")"
  curl -sSfL --retry 8 --retry-delay 30 -C - -o "$f" "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/123acf1cc74bccbb9bfff8ac1ee72edc08c2341d/$f" || { echo "FAIL $f"; continue; }
  [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sha" ] && echo "OK $f" || echo "BADSHA $f"
done
echo DONE
