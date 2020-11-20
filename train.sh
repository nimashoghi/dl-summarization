#!/bin/bash

python train.py --gpus -1 --accelerator ddp --max_epochs 10 --terminate_on_nan
