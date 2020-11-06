#%%
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", fast=True)
tokenizer

# %%
tokenizer(
    "what is the script?",
    "hello world",
    return_tensors="pt",
    return_token_type_ids=True,
)
