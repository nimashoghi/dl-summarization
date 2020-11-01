import argparse

import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    get_linear_schedule_with_warmup,
)

from data import BigPatentDataModule


class SummarizerModule(nn.Module):
    model: BartForConditionalGeneration

    def __init__(self):
        super(SummarizerModule, self).__init__()

        self.model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-large-cnn"
        )

    def forward(self, input):
        input_ids = input["input_ids"]
        input_mask = input["input_mask"]

        output_ids = input["output_ids"]
        output_mask = input["output_mask"]

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

        self.model = SummarizerModule()

    def training_step(self, batch, batch_idx):
        output = self.model(batch)

        return dict(loss=output["loss"], log=dict(train_loss=output["loss"]))

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)

        return dict(val_loss=output["loss"], log=dict(val_loss=output["loss"]))

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        t_total = (
            (10000 // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )

        return [optimizer], [scheduler]


#%%
args_dict = dict(
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=4,
    eval_batch_size=4,
    num_train_epochs=2,
    gradient_accumulation_steps=8,
    n_gpu=2,
    resume_from_checkpoint=None,
    fp_16=True,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level="O1",  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)


args = argparse.Namespace(**args_dict)
train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision=16 if args.fp_16 else 32,
    resume_from_checkpoint=args.resume_from_checkpoint,
    gradient_clip_val=args.max_grad_norm,
    fast_dev_run=True,
    terminate_on_nan=True,
)

#%%
if __name__ == "__main__":
    from pytorch_lightning.loggers import WandbLogger

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = Summarizer(args, tokenizer)
    data_module = BigPatentDataModule(tokenizer)

    trainer = pl.Trainer(
        **train_params,
        logger=WandbLogger(name="bart-summarization"),
    )
    trainer.fit(model, data_module)
