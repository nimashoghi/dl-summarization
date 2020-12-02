#%%
import torch
from rouge_score import rouge_scorer

from summarization.models.blenderbot import BlenderbotSummarizer

#%%
model = BlenderbotSummarizer()
#%%
model.generate_text(
    "When are you allowed to issue a preliminary injunction in a patent case?"
)

#%%

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


#%%
import pandas as pd

df = pd.read_csv("oa_rejection.csv")
df

#%%
# og_model: PegasusSummarizer = PegasusSummarizer(pretrained_name="google/pegasus-large")
# model: LongformerPegasusSummarizer = LongformerPegasusSummarizer.load_from_checkpoint(
#     # "/workspaces/summarization-remote/checkpoints-longformer_pegasus-tldr_legal/epoch=124.ckpt",
#     # "/workspaces/summarization-remote/checkpoints-bigpatent/checkpoints-best/epoch=80.ckpt"
#     # "/workspaces/summarization-remote/checkpoints-longformer_pegasus-big_patent/last.ckpt"
#     "/workspaces/summarization-remote/checkpoints-bigpatent/checkpoints-best/epoch=78.ckpt"
#     # "/workspaces/summarization-remote/checkpoints-best-longformer_pegasus-tldr_legal/epoch=113.ckpt"
# )

og_model = og_model.to("cuda:0")
model = model.to("cuda:1")

#%%
def generate_text(model, text, max_length=6144, device="cuda:0"):
    device = torch.device(device)
    input = model.tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
        return_attention_mask=True,
        return_tensors="pt",
    )
    beam_outputs = model.generate(
        input["input_ids"].to(device),
        attention_mask=input["attention_mask"].to(device),
        max_length=256,
        num_beams=5,
        repetition_penalty=5.0,
        # length_penalty=0.85,
        # num_return_sequences=3,
        early_stopping=True,
    ).cpu()
    output = [
        model.tokenizer.decode(beam_output, skip_special_tokens=True)
        for beam_output in beam_outputs
    ]
    return output[0]


#%%
import random

CPC_CODES = ["a", "b", "c", "d", "e", "f", "g", "h", "y"]
items = [
    item
    for cpc_code in CPC_CODES
    for item in BigPatentDataset.read_data("test", cpc_code)
]
random.shuffle(items)
d = iter(items)

# %%

sample_data = next(d)

sample_data["publication_number"]
description = sample_data["description"]
abstract = sample_data["abstract"]
abstract

#%%
generated_og = generate_text(og_model, description, max_length=1024, device="cuda:0")
generated_og, scorer.score(abstract, generated_og)

#%%
generated = generate_text(model, description, device="cuda:1")
generated, scorer.score(abstract, generated)

# %%
