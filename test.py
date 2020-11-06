#%%
from transformers import T5Tokenizer
from model import Summarizer


tokenizer = T5Tokenizer.from_pretrained("t5-small")
model: Summarizer = Summarizer.load_from_checkpoint(
    "lightning_logs/version_10/checkpoints/epoch=1.ckpt", tokenizer=tokenizer
)
model

# %%
with open("input.txt", "r") as f:
    text = f.read()

model.generate_test(text, max_length=512)
