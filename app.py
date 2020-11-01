#%%
from transformers import LongformerModel, LongformerTokenizer
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)

model = LongformerModel.from_pretrained(
    "allenai/longformer-large-4096", return_dict=True
)
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096")
model, tokenizer

# %%
input_ids = tokenizer.encode(
    "This is a sentence from [MASK] training data", return_tensors="pt"
)
input_ids
# %%
from readData import readData

data = list(readData(".", "train", "d"))

# %%

x = data[0]
tokenizer.encode(
    x["abstract"],
    return_tensors=TensorType.PYTORCH,
    max_length=1024,
    truncation=TruncationStrategy.LONGEST_FIRST,
    padding=PaddingStrategy.MAX_LENGTH,
)
description = tokenizer.encode(x["description"], return_tensors=TensorType.PYTORCH)


model(description)
