#%%
import torch
from rouge_score import rouge_scorer

from summarization.data.big_patent import BigPatentDataset
from summarization.data.tldr_legal import TLDRLegalDataset
from summarization.models.longformer_pegasus import LongformerPegasusSummarizer
from summarization.models.pegasus import PegasusSummarizer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


#%%
og_model: PegasusSummarizer = PegasusSummarizer()
model: LongformerPegasusSummarizer = LongformerPegasusSummarizer.load_from_checkpoint(
    "/workspaces/summarization-remote/checkpoints-tldr/checkpoints/last.ckpt",
    # "/workspaces/summarization-remote/checkpoints-best/epoch=80.ckpt",
)

og_model.cuda()
model.cuda()

#%%
def generate_text(model, text, max_length=6144):
    input = model.tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
        return_attention_mask=True,
        return_tensors="pt",
    )
    beam_outputs = model.generate(
        input["input_ids"].cuda(),
        attention_mask=input["attention_mask"].cuda(),
        max_length=256,
        num_beams=5,
        repetition_penalty=2.0,
        # length_penalty=0.85,
        # num_return_sequences=3,
        early_stopping=True,
    )
    output = [
        model.tokenizer.decode(beam_output, skip_special_tokens=True)
        for beam_output in beam_outputs
    ]
    return output[0]


#%%
# d = BigPatentDataset.read_data("test", "a")
# d
d = iter(TLDRLegalDataset())
# %%
import itertools

sample_data = next(d)

description = sample_data["description"]
abstract = sample_data["abstract"]
abstract

#%%
generated_og = generate_text(og_model, description, max_length=1024)
generated_og, scorer.score(abstract, generated_og)

#%%
generated = generate_text(model, description)
generated, scorer.score(abstract, generated)

# %%
import itertools

sum_og = {}
sum_us = {}
count = 0

for sample_data in itertools.islice(d, 100):
    count += 1
    description = sample_data["description"]
    abstract = sample_data["abstract"]
    generated = generate_text(og_model, description, max_length=1024)
    for metric, value in scorer.score(abstract, generated).items():
        if metric not in sum_og:
            sum_og[metric] = 0

        sum_og[metric] += value.fmeasure

    generated = generate_text(model, description)
    for metric, value in scorer.score(abstract, generated).items():
        if metric not in sum_us:
            sum_us[metric] = 0

        sum_us[metric] += value.fmeasure

#%%
for metric in set(*sum_og.keys(), *sum_us.keys()):
    average_us = sum_us[metric] / count
    average_og = sum_og[metric] / count
    print(f"[{metric}]: {average_us} us vs. {average_og} them")
