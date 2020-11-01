#%%
from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration

tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
model = ProphetNetForConditionalGeneration.from_pretrained(
    "microsoft/prophetnet-large-uncased"
)

#%%
ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
inputs = tokenizer(
    [ARTICLE_TO_SUMMARIZE], max_length=4096, truncation=True, return_tensors="pt"
)
summary_ids = model.generate(
    inputs["input_ids"], num_beams=4, max_length=5, early_stopping=True
)
print(
    [
        tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for g in summary_ids
    ]
)
