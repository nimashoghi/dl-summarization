from transformers import T5ForConditionalGeneration, T5Tokenizer

from summarization.models.base import SummarizerBase


class T5Summarizer(SummarizerBase):
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer

    def __init__(self, *args, **kwargs):
        super(T5Summarizer, self).__init__(
            model_cls=T5ForConditionalGeneration,
            tokenizer_cls=T5Tokenizer,
            pretrained_name="t5-small",
            *args,
            **kwargs
        )
