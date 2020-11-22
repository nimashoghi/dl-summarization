from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

import torch
from summarization.models.base import SummarizerBase
from summarization.util import freeze_params
from torch import Tensor, nn
from transformers import (
    LongformerSelfAttention,
    PegasusConfig,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)


class LongformerPegasusForConditionalGeneration(PegasusForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if config.attention_mode == "n2":
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.model.encoder.layers):
                layer.self_attn = LongformerSelfAttentionForPegasus(config, layer_id=i)


class LongformerPegasusConfig(PegasusConfig):
    def __init__(
        self,
        attention_window: List[int] = None,
        attention_dilation: List[int] = None,
        autoregressive: bool = False,
        attention_mode: str = "tvm",
        **kwargs
    ):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        assert self.attention_mode in ["tvm", "sliding_chunks", "n2"]


class LongformerSelfAttentionForPegasus(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.embed_dim = config.d_model
        self.longformer_self_attn = LongformerSelfAttention(config, layer_id=layer_id)
        self.output = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        query,
        *args,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
        need_weights=False,
        output_attentions=False,
        head_mask=None,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        outputs = self.longformer_self_attn(
            query.transpose(
                0, 1
            ),  # LongformerSelfAttention expects (bsz, seqlen, embd_dim)
            # attention_mask=key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=1) * -1,
            attention_mask=(~key_padding_mask) * 1,
        )

        attn_output = self.output(outputs[0].transpose(0, 1))

        return (
            (attn_output,) + outputs[1:] if len(outputs) == 2 else (attn_output, None)
        )


class LongformerPegasusSummarizer(SummarizerBase):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--freeze_embeddings", type=bool, default=True)
        parser.add_argument("--freeze_decoder", type=bool, default=True)
        return SummarizerBase.add_model_specific_args(parser)

    model: LongformerPegasusForConditionalGeneration
    tokenizer: PegasusTokenizer

    def __init__(self, *args, **kwargs):
        kwargs_new = dict(
            model_cls=LongformerPegasusForConditionalGeneration,
            tokenizer_cls=PegasusTokenizer,
            pretrained_name="/workspaces/summarization-remote/dlt-summarization/converted-models/longformer-pegasus",
            input_length=8192,
            output_length=256,
            return_attention_mask=True,
        )
        kwargs_new.update(kwargs)
        super(LongformerPegasusSummarizer, self).__init__(*args, **kwargs_new)

        freeze_params(self.model.model.decoder)
        # for i, layer in enumerate(self.model.model.encoder.layers):
        #     if i == len(self.model.model.encoder.layers) - 1:
        #         continue
        #     freeze_params(layer)

    # def on_train_epoch_start(self) -> None:
    #     index = random.randint(0, len(self.model.model.encoder.layers) - 1)

    #     for i, layer in enumerate(self.model.model.encoder.layers):
    #         if i == index:
    #             continue
    #         freeze_params(layer, requires_grad=i == index)
