if __name__ == "__main__":
    from util import init_model_from_args

    model, trainer, data = init_model_from_args()

    trainer.test(model, datamoule=data)
