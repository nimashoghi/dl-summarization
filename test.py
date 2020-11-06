#%%
from models.bart import BartSummarizer

# %%
with open("input.txt", "r") as f:
    text = f.read()

#%%
model: BartSummarizer = BartSummarizer({})


#%%

model.generate_test(text, max_length=64)

#%%
model: BartSummarizer = BartSummarizer.load_from_checkpoint(
    "lightning_logs/version_55/checkpoints/epoch=0.ckpt"
)
model.generate_test(text, max_length=1024)
