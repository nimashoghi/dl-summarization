#!/bin/bash

gdown "https://drive.google.com/uc?export=download&id=1J3mucMFTWrgAYa3LuBZoLRR3CzzYD3fa"
tar xvzf bigPatentData.tar.gz
cd bigPatentData
tar xzvf test.tar.gz
tar xzvf train.tar.gz
tar xzvf val.tar.gz
