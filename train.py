#%%
from argparse import ArgumentParser

from pytorch_lightning import Trainer

from data import BigPatentDataModule
from models.prophetnet import ProphetNetSummarizer


#%%
if __name__ == "__main__":
    parser = ArgumentParser()
    parser = ProphetNetSummarizer.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    trainer = Trainer.from_argparse_args(args)

    model = ProphetNetSummarizer(**vars(args))
    data = BigPatentDataModule(model.tokenizer, batch_size=model.hparams.batch_size)

    if trainer.auto_lr_find or trainer.auto_scale_batch_size:
        trainer.tune(model, data)
    else:
        trainer.fit(model, data)
