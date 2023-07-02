# hi
import self_attention
from self_attention import SelfAttentionHead
import torch
import numpy as np

if __name__ == '__main__':

    query = torch.rand([3, 24, 512])
    key = torch.rand([3, 24, 512])
    value = torch.rand([3, 24, 512])

    head = SelfAttentionHead(512, 1024, 512)

    print(head.forward(query, key, value).size())

