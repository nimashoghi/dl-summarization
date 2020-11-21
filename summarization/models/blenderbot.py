from transformers import BlenderbotForConditionalGeneration, BlenderbotSmallTokenizer

from summarization.models.base import SummarizerBase


class BlenderbotSummarizer(SummarizerBase):
    model: BlenderbotForConditionalGeneration
    tokenizer: BlenderbotSmallTokenizer

    def __init__(self, *args, **kwargs):
        super(BlenderbotSummarizer, self).__init__(
            model_cls=BlenderbotForConditionalGeneration,
            tokenizer_cls=BlenderbotSmallTokenizer,
            pretrained_name="facebook/blenderbot-90M",
            *args,
            **kwargs
        )
