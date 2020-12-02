#!/bin/bash

python train.py --model longformer_pegasus --datamodule tldr_legal --gpus -1 --accelerator ddp --batch_size 1 --limit_train_batches 88 --limit_val_batches 32 --val_check_interval 88 --accumulate_grad_batches 16 --resume_from_checkpoint /workspaces/summarization-remote/checkpoints-bigpatent/checkpoints-best/epoch=78.ckpt --max_epochs 130

python train.py --model longformer_pegasus --gpus -1 --accelerator ddp --batch_size 1 --limit_train_batches 256 --limit_val_batches 32 --val_check_interval 128 --accumulate_grad_batches 16 --resume_from_checkpoint /workspaces/summarization-remote/checkpoints-bigpatent/checkpoints-best/epoch=78.ckpt --max_epochs 160
