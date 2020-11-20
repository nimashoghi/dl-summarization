import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import AutoTokenizer

from longformer.longformer_encoder_decoder import (
    LongformerEncoderDecoderConfig,
    LongformerEncoderDecoderForConditionalGeneration,
)


class LongformerSummarizer(pl.LightningModule):
    model: LongformerEncoderDecoderForConditionalGeneration
    tokenizer: AutoTokenizer

    def __init__(self, hparams):
        super(LongformerSummarizer, self).__init__()

        self.hparams = hparams

        config = LongformerEncoderDecoderConfig.from_pretrained(
            "/workspaces/summarization-remote/pretrained/longformer-encdec-base-16384"
        )
        config.attention_dropout = 0.1
        config.gradient_checkpointing = True
        config.attention_mode = "sliding_chunks"
        config.attention_window = [512] * config.encoder_layers
        self.model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(
            "/workspaces/summarization-remote/pretrained/longformer-encdec-base-16384",
            config=config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/workspaces/summarization-remote/pretrained/longformer-encdec-base-16384",
            fast=True,
        )

    def forward(self, input):
        input_ids = input["input_ids"]
        input_mask = input["input_mask"]

        output_ids = input["output_ids"]
        output_mask = input["output_mask"]
        # output_ids[output_ids == self.tokenizer.pad_token_id] = -100

        return self.model(
            input_ids,
            attention_mask=input_mask,
            labels=output_ids,
            decoder_attention_mask=output_mask,
            return_dict=True,
        )

    def generate_test(self, text, max_length=64, **kwargs):
        input = self.tokenizer(
            text,
            max_length=16384,
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
