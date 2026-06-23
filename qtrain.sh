python qtrain.py \
  --scale 2 \
  --channel_nums 32 \
  --num_blocks 5 \
  --epochs 150 \
  --batch_size 64 \
  --lr 2e-3 \
  --minlr 5e-5 \
  --num_workers 8 \
  --save_dir "./checkpoints" \
  --device "cuda" \
  --in_channels 3 \
  --patch_size 144\
  --warmup_epochs 10 \
  --ema_decay 0.999 \
  --wbits 4 \
  --abits 4
# Optional: --pretrained_fp "./checkpoints/DPSR_x2_0514_2125.pth"
