#%%
from models.prophetnet import ProphetNetSummarizer

# %%
with open("input.txt", "r") as f:
    text = f.read()

#%%
model: ProphetNetSummarizer = ProphetNetSummarizer()
model.generate_test(text, max_length=512)

#%%
model: ProphetNetSummarizer = ProphetNetSummarizer.load_from_checkpoint(
    "/workspaces/summarization-remote/lightning_logs/version_86/checkpoints/epoch=1.ckpt"
)
model.generate_test(f"Summarize {text}", max_length=64)
