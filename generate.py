#%%
from models.prophetnet import ProphetNetSummarizer

# %%
with open("input.txt", "r") as f:
    text = f.read()

#%%
model: ProphetNetSummarizer = ProphetNetSummarizer()
model.generate_text(text, input_max_length=64, outupt_max_length=64)

#%%
model: ProphetNetSummarizer = ProphetNetSummarizer.load_from_checkpoint(
    "/workspaces/summarization-remote/lightning_logs/version_86/checkpoints/epoch=1.ckpt"
)
model.generate_text(f"Summarize {text}", max_length=64)
