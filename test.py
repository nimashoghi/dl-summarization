#%%
from models.longformer import LongformerSummarizer

# %%
with open("input.txt", "r") as f:
    text = f.read()

#%%
model: LongformerSummarizer = LongformerSummarizer({})
model.generate_test(text, max_length=512)

#%%
model: LongformerSummarizer = LongformerSummarizer.load_from_checkpoint(
    "lightning_logs/version_22/checkpoints/epoch=0.ckpt"
)
model.generate_test(text, max_length=512)
