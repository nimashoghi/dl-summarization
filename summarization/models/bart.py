from transformers import BartForConditionalGeneration, BartTokenizer

from summarization.models.base import SummarizerBase


class BartSummarizer(SummarizerBase):
    model: BartForConditionalGeneration
    tokenizer: BartTokenizer

    def __init__(self, *args, **kwargs):
        super(BartSummarizer, self).__init__(
            model_cls=BartForConditionalGeneration,
            tokenizer_cls=BartTokenizer,
            pretrained_name="facebook/bart-base",
            *args,
            **kwargs
        )
