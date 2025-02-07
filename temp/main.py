import time
import numpy as np
import torch
from torch.utils.cpp_extension import load

cuda_module = load(name="add2",
                   sources=["add2.cpp", "add2.cu"],
                   verbose=True)

# c = a + b (shape: [n])
n = 1024 * 1024
a = torch.rand(n, device="cuda_programming:0")
b = torch.rand(n, device="cuda_programming:0")
cuda_c = torch.rand(n, device="cuda_programming:0")

def run_cuda():
    cuda_module.torch_launch_add2(cuda_c, a, b, n)
    return cuda_c

def run_torch():
    # return None to avoid intermediate GPU memory application
    # for accurate time statistics
    a + b
    return None

run_cuda() #一个是跑cuda算子
run_torch() #一个是直接跑torch


from transformers import Qwen2Model