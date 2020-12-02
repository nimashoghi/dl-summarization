# Legal Text Summarization

- [Legal Text Summarization](#legal-text-summarization)
  - [Getting the BIGPATENT Dataset](#getting-the-bigpatent-dataset)
  - [Converting Pretrained PEGASUS to PEGASUS-Longformer](#converting-pretrained-pegasus-to-pegasus-longformer)
  - [Fine-Tuning (Training)](#fine-tuning-training)
  - [Finding the Best Learning Rate](#finding-the-best-learning-rate)

## Getting the BIGPATENT Dataset
To get the BIGPATENT dataset, `cd` into the `datasets` directory and run the `get-dataset.sh` script. This will download the BIGPATENT dataset and extract the train, test, and validation archives. You can also run these commands manually (but please make sure you have `gdown` installed from pip).
```bash
gdown "https://drive.google.com/uc?export=download&id=1J3mucMFTWrgAYa3LuBZoLRR3CzzYD3fa"
tar xvzf bigPatentData.tar.gz
cd bigPatentData
tar xzvf test.tar.gz
tar xzvf train.tar.gz
tar xzvf val.tar.gz
```
When training, please make sure the directory variable (`DATASET_DIR` in `summarization/data/big_patent.py`) is set to the directory for your extracted dataset.

## Converting Pretrained PEGASUS to PEGASUS-Longformer
To convert the pretrained `google/pegasus-big_patent` model to our PEGASUS-Longformer model, please use the `scripts/convert-pegasus-to-longformer.py` script. The arguments for this command are shown below. If you do not provide any arguments, it will output your new model to `./converted-models/longformer-pegasus/`.

## Fine-Tuning (Training)

## Finding the Best Learning Rate
To find the best learning rate for a specific network, append the `--auto_lr_find` flag to your the training command as shown above. Then, the trainer will try progressively increasing the learning rate until it finds the optimal one.
