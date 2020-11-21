#!/bin/bash

python train.py longformer_pegasus --gpus -1 --accelerator ddp --terminate_on_nan --batch_size 1 --limit_train_batches 100 --limit_val_batches 25
