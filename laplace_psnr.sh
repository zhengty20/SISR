python laplace_psnr.py \
  --checkpoint "./checkpoints/DPSR_x4_0815_1611.pth" \
  --scale 4\
  --channel_nums 32 \
  --subnet_channels 16 \
  --num_blocks 5 \
  --in_channels 3 \
  --plot_range 1 24 \
  --device cuda
