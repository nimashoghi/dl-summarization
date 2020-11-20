from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import (
    ProphetNetForConditionalGeneration,
    ProphetNetTokenizer,
    get_linear_schedule_with_warmup,
)


class ProphetNetSummarizer(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument(
            "--learning_rate", "-lr", type=float, default=1.9054607179632464e-05
        )
        return parser

    model: ProphetNetForConditionalGeneration
    tokenizer: ProphetNetTokenizer

    def __init__(self, *args, **kwargs):
        super(ProphetNetSummarizer, self).__init__()

        self.save_hyperparameters()
        print(self.hparams)

        self.model = ProphetNetForConditionalGeneration.from_pretrained(
            "microsoft/prophetnet-large-uncased-cnndm"
        )
        self.tokenizer = ProphetNetTokenizer.from_pretrained(
            "microsoft/prophetnet-large-uncased-cnndm"
        )
        # self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input):
        input_ids = input["input_ids"]
        input_mask = input["input_mask"]

        output_ids = input["output_ids"]
        output_mask = input["output_mask"]
        # output_ids[output_ids == self.tokenizer.pad_token_id] = -100

        return self.model(
            input_ids,
            attention_mask=input_mask,
            decoder_attention_mask=output_mask,
            labels=output_ids,
            return_dict=True,
        )

    def generate_test(self, text, max_length=64, **kwargs):
        input = self.tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        )
        beam_outputs = self.generate(
            input["input_ids"],
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            num_return_sequences=5,
            early_stopping=True,
            **kwargs
        )
        for i, beam_output in enumerate(beam_outputs):
            print(
                "{}: {}".format(
                    i, self.tokenizer.decode(beam_output, skip_special_tokens=True)
                )
            )

    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self(batch)

        self.log(
            "train_loss", output["loss"], on_step=True, on_epoch=True, prog_bar=True
        )
        return dict(loss=output["loss"])

    def validation_step(self, batch, batch_idx):
        output = self(batch)

        self.log("val_loss", output["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return dict(val_loss=output["loss"])

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        return optimizer
        # scheduler = get_linear_schedule_with_warmup(optimizer, 500, 5500)

        # return [optimizer], [scheduler]
