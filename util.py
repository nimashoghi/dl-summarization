from argparse import ArgumentParser

from pytorch_lightning import Trainer

from data import BigPatentDataModule
from models.longformer_pegasus import LongformerPegasusSummarizer
from models.pegasus import PegasusSummarizer


def _get_model(model_name: str):
    if model_name == "pegasus":
        return PegasusSummarizer
    elif model_name == "longformer_pegasus":
        return LongformerPegasusSummarizer
    else:
        raise Exception(f"Model {model_name} not found!")


def init_model_from_args():
    parser = ArgumentParser()
    parser.add_argument("model_name", default="pegasus", type=str, help="model name")
    args, _ = parser.parse_known_args()

    model_cls = _get_model(args.model_name)

    parser = Trainer.add_argparse_args(parser)
    parser = model_cls.add_model_specific_args(parser)
    args = parser.parse_args()

    trainer = Trainer.from_argparse_args(args)
    model = model_cls(**vars(args))
    data = BigPatentDataModule(model.hparams, model.tokenizer)

    return model, trainer, data
