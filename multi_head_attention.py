import torch
from torch import Tensor
import torch.nn as nn
from self_attention import SelfAttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, number_of_heads: int, input_dimension: int, queries_keys_dimension: int, values_dimension: int):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(input_dimension, queries_keys_dimension, values_dimension)
                                    for _ in range(number_of_heads)])
        self.feed_forward_layer = nn.Linear(number_of_heads*values_dimension, input_dimension)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        multi_head_result = torch.cat([head(query, key, value) for head in self.heads], dim=-1)

        processed_multi_head_result = self.feed_forward_layer(multi_head_result)

        return processed_multi_head_result

