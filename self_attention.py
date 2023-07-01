'''
sources:
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51
https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
https://www.youtube.com/watch?v=U0s0f995w14&ab_channel=AladdinPersson

'''

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor


# PARAMS
# queries - tensor of shape (batch_size, sequence_length, number_of features_for_queries)
# keys - tensor of shape (batch_size, sequence_length, number_of features_for_keys)
# Note: number_of features_for_keys = number_of features_for_queries
# values - tensor of shape (batch_size, sequence_length, number_of features_for_values)
def scaled_dot_product_attention(queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
    matrix_mult_of_queries_and_keys = queries.bmm(keys.transpose(1, 2))
    scalar_for_gradient_stability = queries.shape(-1) ** (1/2)
    matrix_multiplication_adjusted_by_scalar = matrix_mult_of_queries_and_keys / scalar_for_gradient_stability

    softmax_scaled_matrix_multiplication = f.softmax(matrix_multiplication_adjusted_by_scalar, dim=-1)

    scaled_dot_product_attention_result = softmax_scaled_matrix_multiplication.bmm(values)

    return scaled_dot_product_attention_result


class SelfAttentionHead(nn.Module):
    def __init__(self, input_dimension: int, queries_keys_dimension: int, values_dimension: int):
        super(SelfAttentionHead, self).__init__()
        self.query_weights = nn.Linear(input_dimension, queries_keys_dimension)
        self.key_weights = nn.Linear(input_dimension, queries_keys_dimension)
        self.value_weights = nn.Linear(input_dimension, values_dimension)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        weighted_query = self.query_weights(query)
        weighted_key = self.key_weights(key)
        weighted_value = self.key_weights(value)

        attention = scaled_dot_product_attention(weighted_query, weighted_key, weighted_value)

        return attention
