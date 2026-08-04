python final_test.py \
  --checkpoint "./checkpoints/DPSR_x4_0804_1255.pth" \
  --scale 4 \
  --channel_nums 32 \
  --subnet_channels 16 \
  --num_blocks 5 \
  --in_channels 3 \
  --arm_threshold 20 \
  --arm_subnet_threshold 5 \
  --device cuda
