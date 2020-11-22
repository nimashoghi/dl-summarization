#!/bin/bash

python train.py --model longformer_pegasus --gpus -1 --accelerator ddp --terminate_on_nan --batch_size 1 --limit_train_batches 256 --limit_val_batches 32 --val_check_interval 128 --accumulate_grad_batches 16
