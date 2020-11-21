#%%
from data import BigPatentDataset
from models.pegasus import PegasusSummarizer

# %%
sample_data = next(BigPatentDataset.read_data("test", "a"))
sample_data

#%%
description = sample_data["description"]
abstract = sample_data["abstract"]

#%%
model: PegasusSummarizer = PegasusSummarizer()


#%%
model.generate_text(description)

#%%
abstract


#%%
model: PegasusSummarizer = PegasusSummarizer.load_from_checkpoint(
    "/workspaces/summarization-remote/lightning_logs/version_86/checkpoints/epoch=1.ckpt"
)
model.generate_text(f"Summarize {text}", max_length=64)
