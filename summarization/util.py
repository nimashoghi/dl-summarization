from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from summarization.data import BigPatentDataModule


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


def init_model_from_args():
    parser = ArgumentParser()
    parser.add_argument("model_name", default="pegasus", type=str, help="model name")
    parser = Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args()

    model_cls = _get_model(args.model_name)

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
    data = BigPatentDataModule(model.hparams, model.tokenizer)

    return model, trainer, data
