#!/bin/bash
set -uo pipefail
ROOT="/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/third_party/CORAL/pretrained"

dl_repo() {
  local repo="$1" local dest="$2" local skip_pattern="$3"
  mkdir -p "$dest"
  local files
  files=$(curl -s "https://huggingface.co/api/models/${repo}" \
    | /usr/bin/python3 -c "
import json,sys
d=json.load(sys.stdin)
for s in d.get('siblings',[]):
    p=s['rfilename']
    if '${skip_pattern}' and p.startswith('${skip_pattern}'): continue
    print(p)
")
  for f in $files; do
    local out="$dest/$f"
    if [ -s "$out" ]; then echo "SKIP $repo/$f"; continue; fi
    mkdir -p "$(dirname "$out")"
    curl -sSL --fail --retry 3 --retry-delay 2 \
      -H "Authorization: Bearer ${HF_TOKEN:-}" \
      "https://huggingface.co/${repo}/resolve/main/$f" -o "$out.part"
    if [ -s "$out.part" ]; then mv "$out.part" "$out"; echo "OK   $repo/$f"; else rm -f "$out.part"; echo "FAIL $repo/$f"; fi
  done
}

dl_repo "YuankaiLuo/SimVLA-LIBERO" "$ROOT/SimVLA-LIBERO" ""
dl_repo "HuggingFaceTB/SmolVLM-500M-Instruct" "$ROOT/SmolVLM-500M-Instruct" "onnx/"

echo "=== done ==="
du -sh "$ROOT"/* 2>/dev/null
