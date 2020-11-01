import argparse

import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
)

from data import BigPatentDataModule


class SummarizerModule(nn.Module):
    def __init__(self, tokenizer: BartTokenizer):
        super(SummarizerModule, self).__init__()

        self.tokenizer = tokenizer
        self.model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-large-cnn"
        )

    def forward(self, input):
        input_ids = input["input_ids"]
        input_mask = input["input_mask"]

        output_ids = input["output_ids"]
        output_mask = input["output_mask"]

        print(
            input_ids.size(), input_mask.size(), output_ids.size(), output_mask.size()
        )

        output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100

        return self.model(
            input_ids,
            attention_mask=input_mask,
            decoder_attention_mask=output_mask,
            labels=output_ids,
        )


class Summarizer(pl.LightningModule):
    tokenizer: BartTokenizer

    def __init__(self, hparams, tokenizer: BartTokenizer):
        super(Summarizer, self).__init__()

        self.hparams = hparams
        self.tokenizer = tokenizer

        self.model = SummarizerModule(self.tokenizer)

    def training_step(self, batch, batch_idx):
        output = self.model(batch)

        return dict(loss=output["loss"], log=dict(train_loss=output["loss"]))

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)

        return dict(val_loss=output["loss"], log=dict(val_loss=output["loss"]))

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        return optimizer


#%%
args_dict = dict(
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=2,
    eval_batch_size=2,
    num_train_epochs=2,
    gradient_accumulation_steps=8,
    n_gpu=2,
    fp_16=True,  # if you want to enable 16-bit training then install apex and set this to true
    max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)


args = argparse.Namespace(**args_dict)
train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision=16 if args.fp_16 else 32,
    gradient_clip_val=args.max_grad_norm,
    fast_dev_run=True,
    terminate_on_nan=True,
)

#%%
if __name__ == "__main__":
    from pytorch_lightning.loggers import WandbLogger

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = Summarizer(args, tokenizer)
    data_module = BigPatentDataModule(tokenizer, batch_size=args.train_batch_size)

    trainer = pl.Trainer(
        **train_params,
        logger=WandbLogger(name="bart-summarization"),
    )
    trainer.fit(model, data_module)
