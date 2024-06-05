# -*- coding: utf-8 -*-

import torch

f = torch.nn.Softmax()
x = torch.Tensor([1, 2, 3])
print(f(x))