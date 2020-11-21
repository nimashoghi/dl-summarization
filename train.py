#%%
from argparse import ArgumentParser

from pytorch_lightning import Trainer

from data import BigPatentDataModule
from util import get_model

#%%
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", default="pegasus", type=str, help="model name")
    args, _ = parser.parse_known_args()

    model_cls = get_model(args.model_name)

    parser = Trainer.add_argparse_args(parser)
    parser = model_cls.add_model_specific_args(parser)
    args = parser.parse_args()

    trainer = Trainer.from_argparse_args(args)
    model = model_cls(**vars(args))
    data = BigPatentDataModule(model.tokenizer, batch_size=model.hparams.batch_size)

    if trainer.auto_lr_find or trainer.auto_scale_batch_size:
        trainer.tune(model, data)
    else:
        trainer.fit(model, data)
