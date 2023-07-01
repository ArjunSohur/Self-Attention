# From Youtube video: https://www.youtube.com/watch?v=U0s0f995w14&ab_channel=AladdinPersson
# This code for self attention was taken from the youtube channel Aladdin Perssons


import torch
import torch.nn as nn


# The nn.Module is a powerful pytorch class.  Here, we write a subclass of nn.Module to inherit its capabilities.
# Since nn.module does most of the heavy listing, we only need to override the forward() method
# as well as create our own initialization method
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dimension = embed_size // heads

        assert (self.head_dimension * heads == embed_size), f"Embed size {embed_size} not a multiple of heads {heads}"

        self.queries = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.keys = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.values = nn.Linear(self.head_dimension, self.head_dimension, bias=False)

        self.fully_connected_output = nn.Linear(heads*self.head_dimension, embed_size)

    def forward(self, query, key, value, mask):
        number_of_examples = query.shape[0]

        queries_length, keys_length, values_length = query.shape[1], key.shape[1], value.shape[1]

        # split embedding into self.head pieces
        values = value.reshape(number_of_examples, self.heads, self.head_dimension)
        keys = key.reshape(number_of_examples, self.heads, self.head_dimension)
        queries = query.reshape(number_of_examples, self.heads, self. head_dimension)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape is (number_of_examples, queries_length, heads_dimension)
        # keys shape is (number_of_examples, keys_length, heads_dimension)
        # energy shape is (number_of_examples, heads, query_length, key_length)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float(-1e20))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            number_of_examples, queries_length, self.heads*self.head_dimension
        )
        # attention shape is (number_of_examples, heads, query_length, key_length)
        # values shape is (number_of_examples, value_length, heads, heads_dimension)
        # We want our output to be (number_of_examples, query_length, heads, heads_dimension)
        # after that, we flatten the last two dimensions

        output = self.fully_connected_output(out)

        return output

