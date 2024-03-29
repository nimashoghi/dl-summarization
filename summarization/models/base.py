from argparse import ArgumentParser

from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)


class SummarizerBase(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument(
            "--learning_rate", "-lr", type=float, default=0.00478630092322638
        )
        return parser

    def __init__(
        self,
        model_cls=None,
        tokenizer_cls=None,
        pretrained_name=None,
        model=None,
        tokenizer=None,
        input_length=512,
        output_length=256,
        beam_size=1,
        *args,
        **kwargs
    ):
        super(SummarizerBase, self).__init__()

        self.save_hyperparameters()

        if model_cls is not None and pretrained_name is not None:
            self.model = model_cls.from_pretrained(pretrained_name)
        elif model is not None:
            self.model = model
        else:
            raise Exception("no model")

        if tokenizer_cls is not None and pretrained_name is not None:
            self.tokenizer = tokenizer_cls.from_pretrained(pretrained_name)
        elif tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            raise Exception("no tokenizer")

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

    def generate_text(self, text, **kwargs):
        input = self.tokenizer(
            text,
            max_length=self.hparams.input_length,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_attention_mask=True,
            return_tensors=TensorType.PYTORCH,
        )
        beam_outputs = self.generate(
            input["input_ids"],
            attention_mask=input["attention_mask"],
            max_length=self.hparams.output_length,
            num_beams=self.hparams.beam_size,
            **kwargs
        )
        return self.tokenizer.decode(beam_outputs[0], skip_special_tokens=True)

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

    def test_step(self, batch, batch_idx):
        output = self(batch)

        self.log(
            "test_loss", output["loss"], on_step=True, on_epoch=True, prog_bar=True
        )
        return dict(test_loss=output["loss"])

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        return optimizer
        # scheduler = get_linear_schedule_with_warmup(optimizer, 500, 5500)

        # return [optimizer], [scheduler]
