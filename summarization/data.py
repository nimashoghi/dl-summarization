import itertools
import gzip
import json
import os
from typing import List, Optional

import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)

DATASET_DIR = "datasets/bigPatentData"
CPC_CODES = ["a", "b", "c", "d", "e", "f", "g", "h", "y"]


class BigPatentDataset(IterableDataset):
    @staticmethod
    def read_data(split_type: str, cpc_code: str):
        for file_name in os.listdir(os.path.join(DATASET_DIR, split_type, cpc_code)):
            with gzip.open(
                os.path.join(DATASET_DIR, split_type, cpc_code, file_name), "r"
            ) as fin:
                for row in fin:
                    yield json.loads(row)

    def __init__(self, split_type: str, cpc_code: str = None):
        self.split_type = split_type
        self.cpc_code = cpc_code

    def __iter__(self):
        assert self.cpc_code is not None

        return iter(BigPatentDataset.read_data(self.split_type, self.cpc_code))


class BigPatentRegularDataset(Dataset):
    def __init__(self, split_type: str, cpc_codes: List[str], take_n=35):
        base_path = os.path.join(DATASET_DIR, split_type)
        file_paths = (
            os.path.join(base_path, cpc_code, file_name)
            for cpc_code in cpc_codes
            for file_name in os.listdir(os.path.join(base_path, cpc_code))
        )
        self.content = [
            json.loads(row)
            for file_path in file_paths
            for row in itertools.islice(iter(gzip.open(file_path, "r")), take_n)
        ]

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        return self.content[index]


class BigPatentDataModule(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer: AutoTokenizer):
        super(BigPatentDataModule, self).__init__()

        self.hparams = hparams
        self.tokenizer = tokenizer
        self.datasets = dict()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            for split_type in ("train", "val"):
                self.datasets[split_type] = BigPatentRegularDataset(
                    split_type=split_type,
                    cpc_codes=["a", "b"],
                    take_n=15 if split_type == "val" else 35,
                )

        if stage == "test":
            for split_type in ("test",):
                self.datasets[split_type] = BigPatentRegularDataset(
                    split_type=split_type, cpc_codes=["a", "b"]
                )

    def batch_collate(self, batch):
        input = self.tokenizer(
            [value["description"] for value in batch],
            max_length=self.hparams.input_length,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors=TensorType.PYTORCH,
        )
        output = self.tokenizer(
            [value["abstract"] for value in batch],
            max_length=self.hparams.output_length,
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

    # def worker_init_fn(self, worker_id: int):
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None:
    #         return

    #     dataset: BigPatentDataset = worker_info.dataset
    #     dataset.cpc_code = CPC_CODES[worker_id]

    def make_dataloader(self, split_type: str):
        return DataLoader(
            self.datasets[split_type],
            batch_size=self.hparams.batch_size,
            collate_fn=self.batch_collate,
            num_workers=2,
            pin_memory=True,
            # worker_init_fn=self.worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader("test")

    def train_dataloader(self) -> DataLoader:
        return self.make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader("val")
