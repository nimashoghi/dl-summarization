#%%
import torch
from rouge_score import rouge_scorer

from summarization.data import BigPatentDataset
from summarization.models.longformer_pegasus import LongformerPegasusSummarizer

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

#%%
og_model: LongformerPegasusSummarizer = LongformerPegasusSummarizer()
model: LongformerPegasusSummarizer = LongformerPegasusSummarizer.load_from_checkpoint(
    "/workspaces/summarization-remote/lightning_logs/version_146/checkpoints/epoch=5-v0.ckpt"
)

og_model.cuda()
model.cuda()

#%%
def generate_text(model, text):
    input = model.tokenizer(
        text,
        max_length=4096,
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
        length_penalty=0.5,
        # num_return_sequences=3,
        early_stopping=True,
    )
    output = [
        model.tokenizer.decode(beam_output, skip_special_tokens=True)
        for beam_output in beam_outputs
    ]
    return output[0]


#%%
d = BigPatentDataset.read_data("test", "a")
d
# %%
sample_data = next(d)

description = sample_data["description"]
abstract = sample_data["abstract"]
abstract

#%%
generated_og = generate_text(og_model, description)
generated_og, abstract, scorer.score(abstract, generated_og)

#%%
generated = generate_text(model, description)
# generated = generated.split(" - ")[0]
generated, abstract, scorer.score(abstract, generated)

# %%
