#!/bin/bash

python train.py longformer_pegasus --gpus -1 --accelerator ddp --terminate_on_nan --batch_size 1 --limit_train_batches 256 --limit_val_batches 32 --val_check_interval 128 --accumulate_grad_batches 16 --resume_from_checkpoint /workspaces/summarization-remote/lightning_logs/version_151/checkpoints/epoch=6-v0.ckpt
