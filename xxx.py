# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


batch_size, seq_length = (2, 10)
seq_length_with_past = seq_length
past_key_values_length = 5

seq_length_with_past = seq_length_with_past + past_key_values_length


position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length)
position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

embed_tokens = nn.Embedding(30, 17, 0)
input_ids = torch.randint(1, 30, (2, 10))
print(input_ids)
inputs_embeds = embed_tokens(input_ids)
print(inputs_embeds.shape)  # (batch_size, seq_len, hidden)

attention_mask = torch.ones((batch_size, seq_length_with_past))
print(attention_mask)

bsz, tgt_len = (batch_size, seq_length)
mask = torch.full((tgt_len, tgt_len), torch.finfo(inputs_embeds.dtype).min)  # 负无穷填充
mask_cond = torch.arange(mask.size(-1))
mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
mask = mask.to(inputs_embeds.dtype)  # [seq_len, seq_len], 上三角是负无穷

if past_key_values_length > 0:
    # [seq_len, seq_len + past_key_values_length], 前past_key_values_length列填充0
    mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=inputs_embeds.dtype), mask], dim=-1)

combined_attention_mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
# print(combined_attention_mask.shape)  # [bsz, 1, seq_len, seq_len + past_key_value_length]
print(combined_attention_mask)


# bsz, src_len = attention_mask.size()
# tgt_len = seq_length
# print(tgt_len, src_len)
# tgt_len = tgt_len if tgt_len is not None else src_len
#
# # [batch_size, 1, seq_len, hidden_size]  1
# expanded_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(inputs_embeds.dtype)
# print(expanded_mask.shape)
# inverted_mask = 1.0 - expanded_mask  # 0
# expanded_attn_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)
# combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
#             )
#
# print(combined_attention_mask)