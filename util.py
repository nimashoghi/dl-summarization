from models.pegasus import PegasusSummarizer


def get_model(model_name: str):
    if model_name == "pegasus":
        return PegasusSummarizer
    else:
        raise Exception(f"Model {model_name} not found!")
