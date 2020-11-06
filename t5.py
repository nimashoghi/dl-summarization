import argparse

import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from data import BigPatentDataModule


class SummarizerModule(nn.Module):
    model: T5ForConditionalGeneration

    def __init__(self, tokenizer: T5Tokenizer):
        super(SummarizerModule, self).__init__()

        self.tokenizer = tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def forward(self, input):
        input_ids = input["input_ids"]
        input_mask = input["input_mask"]

        output_ids = input["output_ids"]
        output_mask = input["output_mask"]

        return self.model(
            input_ids,
            attention_mask=input_mask,
            decoder_attention_mask=output_mask,
            labels=output_ids,
            return_dict=True,
        )


class Summarizer(pl.LightningModule):
    tokenizer: T5Tokenizer

    def __init__(self, hparams, tokenizer: T5Tokenizer):
        super(Summarizer, self).__init__()

        self.hparams = hparams
        self.tokenizer = tokenizer

        self.model = SummarizerModule(self.tokenizer)

    def training_step(self, batch, batch_idx):
        output = self.model(batch)

        self.log(
            "train_loss", output["loss"], on_step=True, on_epoch=True, prog_bar=True
        )
        return dict(loss=output["loss"])

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)

        self.log("val_loss", output["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return dict(val_loss=output["loss"])

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
    train_batch_size=3,
    eval_batch_size=3,
    num_train_epochs=2,
    n_gpu=2,
    seed=42,
)


args = argparse.Namespace(**args_dict)
train_params = dict(
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision=16,
    terminate_on_nan=True,
)

#%%
if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = Summarizer(args, tokenizer)
    data_module = BigPatentDataModule(tokenizer, batch_size=args.train_batch_size)

    trainer = pl.Trainer(**train_params, distributed_backend="ddp")
    trainer.fit(model, data_module)
