#%%
import argparse

import pytorch_lightning as pl

args_dict = dict(
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=2,
    eval_batch_size=2,
    num_train_epochs=1,
    n_gpu=2,
    seed=42,
)


args = argparse.Namespace(**args_dict)
train_params = dict(
    distributed_backend="ddp",
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision=16,
    terminate_on_nan=True,
    # fast_dev_run=True,
    # gpus=0,
)

#%%
if __name__ == "__main__":
    from data import BigPatentDataModule
    from models.bart import BartSummarizer

    model = BartSummarizer(args)
    data_module = BigPatentDataModule(model.tokenizer, batch_size=args.train_batch_size)

    trainer = pl.Trainer(**train_params)
    trainer.fit(model, data_module)
