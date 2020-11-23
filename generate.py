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
    # "/workspaces/summarization-remote/checkpoints-longformer_pegasus-tldr_legal/epoch=124.ckpt",
    # "/workspaces/summarization-remote/checkpoints-bigpatent/checkpoints-best/epoch=80.ckpt"
    # "/workspaces/summarization-remote/checkpoints-longformer_pegasus-big_patent/last.ckpt"
    "/workspaces/summarization-remote/checkpoints-bigpatent/checkpoints-best/epoch=78.ckpt"
)

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
import itertools

sample_data = next(d)

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
import random

# random sampling
samples = random.sample(items, 600)
len(items), len(samples)

#%%
# top length sampling
top_length_samples = sorted(items, key=lambda x: len(x["abstract"]), reverse=True)[
    0:256
]
len(items), len(top_length_samples)

#%%
runs_og = []
runs_us = []
sum_og = {}
sum_us = {}
count = 0

from threading import Thread


def run(sample_data, model, runs, device):
    description = sample_data["description"]
    abstract = sample_data["abstract"]
    generated = generate_text(model, description, max_length=1024, device=device)

    info = dict(
        abstract_length=len(abstract),
        description_length=len(description),
    )

    metrics = scorer.score(abstract, generated)
    runs.append(
        dict(
            **info,
            generated=generated,
            abstract=abstract,
            metrics=metrics,
            generated_length=len(generated),
        )
    )


for sample_data in top_length_samples:
    count += 1
    t1 = Thread(target=run, args=[sample_data, og_model, runs_og, "cuda:0"])
    t1.start()

    t2 = Thread(target=run, args=[sample_data, model, runs_us, "cuda:1"])
    t2.start()

    t1.join()
    t2.join()

    print(".", end="")

#%%
for metric in set([*sum_og.keys(), *sum_us.keys()]):
    average_us = sum_us[metric] / count
    average_og = sum_og[metric] / count
    print(f"[{metric}]: {average_us} us vs. {average_og} them")

#%%
data = dict(
    sum_og=sum_og,
    sum_us=sum_us,
    count=count,
    runs_og=runs_og,
    runs_us=runs_us,
    samples=samples,
)
data
#%%
import pickle

with open("data-nov23-highest-count.bin", "wb") as f:
    pickle.dump(data, f)


#%%
import itertools

runs = list(
    zip(
        sorted(runs_us, key=lambda x: x["abstract_length"], reverse=True),
        sorted(runs_og, key=lambda x: x["abstract_length"], reverse=True),
    )
)

m_us = dict(rouge1=0, rouge2=0, rougeL=0)
m_them = dict(rouge1=0, rouge2=0, rougeL=0)
for r1, r2 in itertools.islice(zip(runs_us, runs_og), 1):
    print(dict(abs=r1["abstract"], r1=r1["generated"], r2=r2["generated"]))
    for metric, value in r1["metrics"].items():
        m_us[metric] += value.fmeasure

    for metric, value in r2["metrics"].items():
        m_them[metric] += value.fmeasure

m_us, m_them

#%%
for metric in set([*m_us.keys(), *m_them.keys()]):
    average_us = m_us[metric] / len(runs)
    average_og = m_them[metric] / len(runs)
    print(f"[{metric}]: {average_us} us vs. {average_og} pegasus")

# %%
