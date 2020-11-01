import gzip
import json
import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from transformers import T5Tokenizer
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
    for cpc_code in os.listdir(os.path.join("bigPatentData", split_type)):
        for file_name in os.listdir(
            os.path.join("bigPatentData", split_type, cpc_code)
        ):
            with gzip.open(
                os.path.join("bigPatentData", split_type, cpc_code, file_name), "r"
            ) as fin:
                for row in fin:
                    yield json.loads(row)


class BigPatentDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer: T5Tokenizer, batch_size=1, sequence_length=1024):
        super(BigPatentDataModule, self).__init__()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer

        self.datasets = dict()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            for split_type in ("train", "val"):
                self.datasets[split_type] = MyDataset(read_data(split_type))

        if stage == "test" or stage is None:
            for split_type in "test":
                self.datasets[split_type] = MyDataset(read_data(split_type))

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
            self.datasets[split_type],
            batch_size=self.batch_size,
            collate_fn=self.batch_collate,
            num_workers=8,
        )

    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader("test")

    def train_dataloader(self) -> DataLoader:
        return self.make_dataloader("val")

    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader("val")
