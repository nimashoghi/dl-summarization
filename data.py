import gzip
import json
import os

import pytorch_lightning as pl
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from transformers import BartTokenizer
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)


class MyDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return iter(self.generator)


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
    def __init__(self, tokenizer: BartTokenizer, batch_size=8, sequence_length=1024):
        super(BigPatentDataModule, self).__init__()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer

    def batch_collate(self, batch):
        input = self.tokenizer(
            [value["description"] for value in batch],
            max_length=self.sequence_length,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors=TensorType.PYTORCH,
        )
        output = self.tokenizer(
            [value["abstract"] for value in batch],
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

    def make_dataloader(self, split_type: str):
        return DataLoader(
            MyDataset(read_data(split_type)),
            batch_size=self.batch_size,
            collate_fn=self.batch_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader("test")

    def train_dataloader(self) -> DataLoader:
        return self.make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader("val")
