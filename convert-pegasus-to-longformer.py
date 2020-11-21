import argparse
import logging
import os

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers.modeling_bart import shift_tokens_right

from summarization.models.longformer_pegasus import (
    LongformerPegasusConfig,
    LongformerPegasusForConditionalGeneration,
    LongformerSelfAttentionForPegasus,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_long_model(
    save_model_to, base_model, tokenizer_name_or_path, attention_window, max_pos
):
    model = PegasusForConditionalGeneration.from_pretrained(base_model)
    tokenizer = PegasusTokenizer.from_pretrained(
        tokenizer_name_or_path, model_max_length=max_pos
    )
    config = LongformerPegasusConfig.from_pretrained(base_model)
    model.config = config

    # in Pegasus attention_probs_dropout_prob is attention_dropout, but LongformerSelfAttention
    # expects attention_probs_dropout_prob, so set it here
    config.attention_probs_dropout_prob = config.attention_dropout
    config.architectures = [
        "LongformerPegasusForConditionalGeneration",
    ]

    N = 0

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs["model_max_length"] = max_pos
    current_max_pos, embed_size = model.model.encoder.embed_positions.weight.shape
    print(max_pos, current_max_pos, embed_size, config.max_position_embeddings)
    assert current_max_pos == config.max_position_embeddings + N

    config.max_encoder_position_embeddings = max_pos
    config.max_decoder_position_embeddings = config.max_position_embeddings
    del config.max_position_embeddings
    max_pos += N  # NOTE: Pegasus has positions 0,1 reserved, so embedding size is max position + N
    assert max_pos >= current_max_pos

    # allocate a larger position embedding matrix for the encoder
    new_encoder_pos_embed = model.model.encoder.embed_positions.weight.new_empty(
        max_pos, embed_size
    )
    # copy position embeddings over and over to initialize the new position embeddings
    k = N
    step = current_max_pos - N
    while k < max_pos - 1:
        new_encoder_pos_embed[
            k : (k + step)
        ] = model.model.encoder.embed_positions.weight[N:]
        k += step
    model.model.encoder.embed_positions.weight.data = new_encoder_pos_embed

    # allocate a larger position embedding matrix for the decoder
    # new_decoder_pos_embed = model.model.decoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # # copy position embeddings over and over to initialize the new position embeddings
    # k = 2
    # step = current_max_pos - 2
    # while k < max_pos - 1:
    #     new_decoder_pos_embed[k:(k + step)] = model.model.decoder.embed_positions.weight[2:]
    #     k += step
    # model.model.decoder.embed_positions.weight.data = new_decoder_pos_embed

    # replace the `modeling_Pegasus.SelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers

    for i, layer in enumerate(model.model.encoder.layers):
        longformer_self_attn_for_Pegasus = LongformerSelfAttentionForPegasus(
            config, layer_id=i
        )

        longformer_self_attn_for_Pegasus.longformer_self_attn.query = (
            layer.self_attn.q_proj
        )
        longformer_self_attn_for_Pegasus.longformer_self_attn.key = (
            layer.self_attn.k_proj
        )
        longformer_self_attn_for_Pegasus.longformer_self_attn.value = (
            layer.self_attn.v_proj
        )

        longformer_self_attn_for_Pegasus.longformer_self_attn.query_global = (
            layer.self_attn.q_proj
        )
        longformer_self_attn_for_Pegasus.longformer_self_attn.key_global = (
            layer.self_attn.k_proj
        )
        longformer_self_attn_for_Pegasus.longformer_self_attn.value_global = (
            layer.self_attn.v_proj
        )

        longformer_self_attn_for_Pegasus.output = layer.self_attn.out_proj

        layer.self_attn = longformer_self_attn_for_Pegasus
    logger.info(f"saving model to {save_model_to}")
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Convert Pegasus to LongPegasus. Replaces Pegasus encoder's SelfAttnetion with LongformerSelfAttention"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/pegasus-big_patent",
        help="The name or path of the base model you want to convert",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="google/pegasus-big_patent",
        help="The name or path of the tokenizer",
    )
    parser.add_argument(
        "--save_model_to",
        default="./converted-models/longformer-pegasus",
        type=str,
        # required=True,
        help="The path to save the converted model",
    )
    parser.add_argument(
        "--attention_window",
        type=int,
        default=1024,
        help="attention window size for longformer self attention (one sided)",
    )
    parser.add_argument(
        "--max_pos", type=int, default=6144, help="maximum encoder positions"
    )
    parser.add_argument(
        "--skip_create", action="store_true", help="skip create long model"
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_model_to):
        os.mkdir(args.save_model_to)

    if not args.skip_create:
        create_long_model(
            save_model_to=args.save_model_to,
            base_model=args.base_model,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            attention_window=args.attention_window,
            max_pos=args.max_pos,
        )

    tokenizer = PegasusTokenizer.from_pretrained(args.save_model_to)
    TXT = "My friends are <mask> but they eat too many carbs."
    model = LongformerPegasusForConditionalGeneration.from_pretrained(
        args.save_model_to
    )
    print(model)
    data = tokenizer(
        [TXT], return_tensors="pt", padding="max_length", max_length=4096 * 2
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    decoder_input_ids = shift_tokens_right(input_ids[:, :5], tokenizer.pad_token_id)
    logits = model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
    )[0]
    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    probs = logits[0, masked_index].softmax(dim=0)
    values, predictions = probs.topk(5)
    print(tokenizer.convert_ids_to_tokens(predictions))


if __name__ == "__main__":
    main()
