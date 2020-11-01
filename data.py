import gzip
import json
import os

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import BartConfig, BartTokenizer
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)


def read_data(split_type: str):
    for cpc_code in os.path.join("bigPatentData", split_type):
        for file_name in os.listdir(
            os.path.join("bigPatentData", split_type, cpc_code)
        ):
            with gzip.open(
                os.path.join("bigPatentData", split_type, cpc_code, file_name), "r"
            ) as fin:
                for row in fin:
                    yield json.loads(row)


class BigPatentDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer: BartTokenizer, batch_size=32, sequence_length=1024):
        super(BigPatentDataModule, self).__init__()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer

    def batch_collate(self, batch):
        input = self.tokenizer(
            [input for input, _ in batch],
            max_length=self.sequence_length,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors=TensorType.PYTORCH,
        )
        output = self.tokenizer(
            [output for _, output in batch],
            max_length=self.sequence_length,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors=TensorType.PYTORCH,
        )
        return dict(
            input_ids=input["input_ids"],
            input_mask=input["attention_mask"],
            output_ids=output["input_ids"],
            output_mask=output["attention_mask"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            read_data("test"),
            batch_size=self.batch_size,
            collate_fn=self.batch_collate,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            read_data("train"),
            batch_size=self.batch_size,
            collate_fn=self.batch_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            read_data("val"),
            batch_size=self.batch_size,
            collate_fn=self.batch_collate,
        )
