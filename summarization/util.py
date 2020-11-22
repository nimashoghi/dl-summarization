from argparse import ArgumentParser

from bs4 import BeautifulSoup
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def strip_html(text: str):
    return BeautifulSoup(text).get_text()


def freeze_params(model, requires_grad=False):
    for par in model.parameters():
        par.requires_grad = requires_grad


def _get_model(model_name: str):
    if model_name == "pegasus":
        from summarization.models.pegasus import PegasusSummarizer

        return PegasusSummarizer
    elif model_name == "longformer_pegasus":
        from summarization.models.longformer_pegasus import LongformerPegasusSummarizer

        return LongformerPegasusSummarizer
    else:
        raise Exception(f"Model {model_name} not found!")


def _get_datamodule(datamodule_name: str):
    if datamodule_name == "big_patent":
        from summarization.data.big_patent import BigPatentDataModule

        return BigPatentDataModule
    elif datamodule_name == "tldr_legal":
        from summarization.data.tldr_legal import TLDRLegalDataModule

        return TLDRLegalDataModule
    else:
        raise Exception(f"Datamodule {datamodule_name} not found!")


def init_model_from_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model", default="pegasus", type=str, help="model name", dest="model_name"
    )
    parser.add_argument(
        "--datamodule",
        default="big_patent",
        type=str,
        help="datamodule name",
        dest="datamodule_name",
    )
    parser = Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args()

    model_cls = _get_model(args.model_name)
    datamodule_cls = _get_datamodule(args.datamodule_name)

    parser = model_cls.add_model_specific_args(parser)
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints/", save_last=True)
    checkpoint_callback2 = ModelCheckpoint(
        dirpath="./checkpoints-best/", save_top_k=2, monitor="val_loss"
    )
    trainer = Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback, checkpoint_callback2]
    )
    model = model_cls(**vars(args))
    data = datamodule_cls(model.hparams, model.tokenizer)

    return model, trainer, data
