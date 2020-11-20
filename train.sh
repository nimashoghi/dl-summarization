#!/bin/bash

python train.py --gpus -1 --accelerator ddp --terminate_on_nan
