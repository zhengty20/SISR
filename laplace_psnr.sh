python laplace_psnr.py \
  --checkpoint "./checkpoints/DPSR_x2_0803_1730.pth" \
  --channel-nums 32 \
  --subnet-channels 16 \
  --num-blocks 5 \
  --in-channels 3 \
  --plot-range 1 24 \
  --device cuda
