from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from summarization.models.base import SummarizerBase


class PegasusSummarizer(SummarizerBase):
    model: PegasusForConditionalGeneration
    tokenizer: PegasusTokenizer

    def __init__(self, *args, **kwargs):
        super(PegasusSummarizer, self).__init__(
            model_cls=PegasusForConditionalGeneration,
            tokenizer_cls=PegasusTokenizer,
            pretrained_name="google/pegasus-big_patent",
            input_length=1024,
            output_length=256,
            *args,
            **kwargs
        )
