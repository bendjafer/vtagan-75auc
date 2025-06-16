#!/bin/bash

# Set dynamic names for the runs
NAME1="ucsd2_weekend_large_run1"
NAME2="ucsd2_weekend_safe_run1"

# Run 1 - Larger batch size, longer training
python3.7 train_video.py \
  --dataset ucsd2 \
  --dataroot data/ucsd2 \
  --model ocr_gan_video \
  --name "$NAME1" \
  --batchsize 8 \
  --isize 256 \
  --num_frames 16 \
  --niter 60 \
  --lr 0.00025 \
  --beta1 0.9 \
  --lr_policy step \
  --lr_decay_iters 20 \
  --grad_clip_norm 1.0 \
  --w_adv 1.0 \
  --w_con 45 \
  --w_lat 1.0 \
  --w_temporal_consistency 0.12 \
  --w_temporal_motion 0.06 \
  --w_temporal_reg 0.015 \
  --use_ucsd_augmentation \
  --ucsd_augmentation moderate \
  --aspect_method maintain_3_2 \
  --device gpu \
  --gpu_ids 2 \
  --workers 16

# Run 2 - Conservative setup
python3.7 train_video.py \
  --dataset ucsd2 \
  --dataroot data/ucsd2 \
  --model ocr_gan_video \
  --name "$NAME2" \
  --batchsize 4 \
  --isize 256 \
  --num_frames 16 \
  --niter 45 \
  --lr 0.0002 \
  --beta1 0.9 \
  --lr_policy step \
  --lr_decay_iters 15 \
  --grad_clip_norm 1.0 \
  --w_adv 1.0 \
  --w_con 50 \
  --w_lat 1.0 \
  --w_temporal_consistency 0.08 \
  --w_temporal_motion 0.04 \
  --w_temporal_reg 0.01 \
  --use_ucsd_augmentation \
  --ucsd_augmentation conservative \
  --aspect_method maintain_3_2 \
  --device gpu \
  --gpu_ids 2 \
  --workers 16
