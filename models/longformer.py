import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import LongformerForMaskedLM, LongformerTokenizer
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)


class LongformerSummarizer(pl.LightningModule):
    model: LongformerForMaskedLM
    tokenizer: LongformerTokenizer

    def __init__(self, hparams):
        super(LongformerSummarizer, self).__init__()

        self.hparams = hparams
        self.model = LongformerForMaskedLM.from_pretrained(
            "allenai/longformer-base-4096"
        )
        self.tokenizer = LongformerTokenizer.from_pretrained(
            "allenai/longformer-base-4096"
        )

    def forward(self, input):
        input_ids = input["input_ids"]
        input_mask = input["input_mask"]

        output_ids = input["output_ids"]

        return self.model(
            input_ids,
            attention_mask=input_mask,
            labels=output_ids,
            return_dict=True,
        )

    def generate_test(self, text, max_length=64, **kwargs):
        input = self.tokenizer(
            text,
            max_length=max_length,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors=TensorType.PYTORCH,
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
