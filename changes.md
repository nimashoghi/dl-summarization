/opt/conda/lib/python3.8/site-packages/transformers/modeling_longformer.py:505

Comment out
```python
local_attn_probs = torch.masked_fill(local_attn_probs, is_index_masked[:, :, None, None], 0.0)
```

modeling_bert:
https://github.com/huggingface/transformers/commit/18ad30c957b92ba1befd446f6e13a209849eb450 apply this
