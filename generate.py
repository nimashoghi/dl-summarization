#%%
from data import BigPatentDataset
from models.longformer_pegasus import LongformerPegasusSummarizer

# %%
sample_data = next(BigPatentDataset.read_data("test", "a"))

description = sample_data["description"]
abstract = sample_data["abstract"]
abstract

#%%
model: LongformerPegasusSummarizer = LongformerPegasusSummarizer()
dir(model.model.model.encoder)

#%%
model.generate_text(description)

#%%
abstract

#%%
model: LongformerPegasusSummarizer = LongformerPegasusSummarizer.load_from_checkpoint(
    "/workspaces/summarization-remote/lightning_logs/version_86/checkpoints/epoch=1.ckpt"
)
model.generate_text(f"Summarize {text}", max_length=64)
