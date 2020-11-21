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
            "--learning_rate", "-lr", type=float, default=1.9054607179632464e-05
        )
        return parser

    def __init__(self, model_cls, tokenizer_cls, pretrained_name: str, *args, **kwargs):
        super(SummarizerBase, self).__init__()

        self.save_hyperparameters()
        print(self.hparams)

        self.model = model_cls.from_pretrained(pretrained_name)
        self.tokenizer = tokenizer_cls.from_pretrained(pretrained_name)

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

    def generate_text(self, text, input_max_length=512, outupt_max_length=64, **kwargs):
        input = self.tokenizer(
            text,
            max_length=input_max_length,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors=TensorType.PYTORCH,
        )
        beam_outputs = self.generate(
            input["input_ids"],
            max_length=outupt_max_length,
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
