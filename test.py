#%%
from models.pegasus import PegasusSummarizer

# %%
with open("input.txt", "r") as f:
    text = f.read()

#%%
model: PegasusSummarizer = PegasusSummarizer()
model.generate_text(text, max_length=512)

#%%
model: PegasusSummarizer = PegasusSummarizer.load_from_checkpoint(
    "/workspaces/summarization-remote/lightning_logs/version_86/checkpoints/epoch=1.ckpt"
)
model.generate_text(f"Summarize {text}", max_length=64)
