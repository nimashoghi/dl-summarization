from typing import Optional

import ndjson
import pytorch_lightning as pl
from summarization.util import strip_html
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)

DATASET_PATH = (
    "/workspaces/summarization-remote/datasets/tldr-legal/tldr-legal-info.ndjson"
)


def get_content(item):
    try:
        text = item["modules"]["fulltext"]["text"]
        summary = item["modules"]["summary"]
        summary_text = summary["text"]
        for key in ("can", "cannot", "must"):
            for point in summary[key]:
                try:
                    description = point["description"]
                    summary_text += f" {description}"
                except KeyError:
                    title = point["attribute"]["title"].lower()
                    description = f"You {key} {title}."
    except:
        return None

    return dict(description=strip_html(text), abstract=strip_html(summary_text))


class TLDRLegalDataset(Dataset):
    def __init__(self, path=DATASET_PATH):
        with open(path, "r") as f:
            self.data = [
                content
                for item in ndjson.load(f)
                if (content := get_content(item)) is not None
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TLDRLegalDataModule(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer: AutoTokenizer):
        super(TLDRLegalDataModule, self).__init__()

        self.hparams = hparams
        self.tokenizer = tokenizer
        self.datasets = dict()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            for split_type in ("train", "val"):
                self.datasets[split_type] = TLDRLegalDataset(split_type)

        if stage == "test" or stage is None:
            for split_type in ("test",):
                self.datasets[split_type] = TLDRLegalDataset(split_type)

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

    def make_dataloader(self, split_type: str):
        return DataLoader(
            self.datasets[split_type],
            batch_size=self.hparams.batch_size,
            collate_fn=self.batch_collate,
            num_workers=8,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader("test")

    def train_dataloader(self) -> DataLoader:
        return self.make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader("val")