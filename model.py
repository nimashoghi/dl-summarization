import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Summarizer(pl.LightningModule):
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer

    def __init__(self, hparams, tokenizer: T5Tokenizer):
        super(Summarizer, self).__init__()

        self.hparams = hparams
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

    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self(batch)

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
