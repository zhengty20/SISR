#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:?Usage: $0 /path/to/DPSR_checkpoint.pth [subnet_channels]}"
subnet_channels="${2:-16}"
cd "$(dirname "$0")"
conda run -n SISR python test.py \
  --checkpoint "$checkpoint" \
  --scale 2 \
  --channel_nums 32 \
  --subnet-channels "$subnet_channels" \
  --num_blocks 5 \
  --in_channels 3 \
  --datasets Set5 Set14 B100 U100 M109 \
  --device cuda
