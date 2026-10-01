#!/bin/bash
set -uo pipefail
REPO="yifengzhu-hf/LIBERO-datasets"
DEST="/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/third_party/CORAL/datasets/metas"
SUITES=(libero_spatial libero_object libero_goal libero_10)
API="https://huggingface.co/api/datasets/${REPO}/tree/main"
BASE="https://huggingface.co/datasets/${REPO}/resolve/main"
LIST=$(mktemp)

mkdir -p "$DEST"
for suite in "${SUITES[@]}"; do
  curl -s "$API/$suite" | /usr/bin/python3 -c "
import json,sys
for f in json.load(sys.stdin):
    if f['path'].endswith('.hdf5'):
        print(f['path'])
" >> "$LIST"
done

echo "Total files to consider: $(wc -l < "$LIST")"

download_one() {
  local path="$1"
  local suite; suite=$(dirname "$path")
  local fname; fname=$(basename "$path")
  local out="$DEST/$suite/$fname"
  mkdir -p "$DEST/$suite"
  if [ -s "$out" ]; then
    echo "SKIP $path"
    return 0
  fi
  curl -sSL --fail --retry 3 --retry-delay 3 \
    -H "Authorization: Bearer ${HF_TOKEN:-}" \
    "$BASE/$path" -o "$out.part"
  if [ -s "$out.part" ]; then
    mv "$out.part" "$out"
    echo "OK   $path"
  else
    rm -f "$out.part"
    echo "FAIL $path"
  fi
}
export -f download_one
export DEST BASE HF_TOKEN

xargs -a "$LIST" -P 8 -I{} bash -c 'download_one "$@"' _ {}

echo "=== done. summary ==="
find "$DEST" -name "*.hdf5" -type f | wc -l
du -sh "$DEST" 2>/dev/null
rm -f "$LIST"
