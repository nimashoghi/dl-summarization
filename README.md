# Legal Text Summarization

## Converting Pretrained PEGASUS to PEGASUS-Longformer
To convert the pretrained `google/pegasus-big_patent` model to our PEGASUS-Longformer model, please use the `scripts/convert-pegasus-to-longformer.py` script. The arguments for this command are shown below. If you do not provide any arguments, it will output your new model to `./converted-models/longformer-pegasus/`.

## Fine-Tuning (Training)

## Finding the Best Learning Rate
To find the best learning rate for a specific network, append the `--auto_lr_find` flag to your the training command as shown above.
