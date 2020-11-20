from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)

from models.base import SummarizerBase


class ProphetNetSummarizer(SummarizerBase):
    @staticmethod
    def add_model_specific_args(parent_parser):
        return SummarizerBase.add_model_specific_args(parent_parser)

    model: ProphetNetForConditionalGeneration
    tokenizer: ProphetNetTokenizer

    def __init__(self, *args, **kwargs):
        super(ProphetNetSummarizer, self).__init__(
            model_cls=ProphetNetForConditionalGeneration,
            tokenizer_cls=ProphetNetTokenizer,
            pretrained_name="microsoft/prophetnet-large-uncased-cnndm",
            *args,
            **kwargs
        )
