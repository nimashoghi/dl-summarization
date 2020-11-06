import gzip
import json
import os
from typing import Optional, Union

import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from transformers import T5Tokenizer
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)


def read_data(split_type: str, skip_n=0, take_n: Union[int, None] = None):
    i = 0
    for cpc_code in os.listdir(os.path.join("bigPatentData", split_type)):
        for file_name in os.listdir(
            os.path.join("bigPatentData", split_type, cpc_code)
        ):
            with gzip.open(
                os.path.join("bigPatentData", split_type, cpc_code, file_name), "r"
            ) as fin:
                for row in fin:
                    index = i - skip_n
                    i += 1

                    if index < 0:
                        continue
                    if take_n is not None and index > take_n:
                        break
                    yield json.loads(row)


class MyDataset(IterableDataset):
    def __init__(self, split_type: str, skip_n=0, take_n: Union[int, None] = None):
        self.split_type = split_type
        self.skip_n = skip_n
        self.take_n = take_n

    def __iter__(self):
        return iter(read_data(self.split_type, skip_n=self.skip_n, take_n=self.take_n))


class BigPatentDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer: T5Tokenizer, batch_size=1, sequence_length=4096):
        super(BigPatentDataModule, self).__init__()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer

        self.datasets = dict()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            for split_type in ("train", "val"):
                self.datasets[split_type] = MyDataset(split_type)

        if stage == "test" or stage is None:
            for split_type in "test":
                self.datasets[split_type] = MyDataset(split_type)

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

    def worker_init_fn(self, worker_id: int):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return

        dataset: MyDataset = worker_info.dataset
        dataset.take_n = 50
        dataset.skip_n += worker_id * dataset.take_n

    def make_dataloader(self, split_type: str):
        return DataLoader(
            self.datasets[split_type],
            batch_size=self.batch_size,
            collate_fn=self.batch_collate,
            num_workers=8,
            pin_memory=True,
            worker_init_fn=self.worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader("test")

    def train_dataloader(self) -> DataLoader:
        return self.make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader("val")
