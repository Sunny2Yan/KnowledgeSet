import time
import torch
import triton
import triton.language as tl


# DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = torch.device("cuda:0")

@triton.jit
def add_kernel(x_ptr,  # Pointer to first input vector.
               y_ptr,  # Pointer to second input vector.
               output_ptr,  # Pointer to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value. 注意：`constexpr` 因此它可以用作形状值。
               ):
    """加法计算内核"""
    # 有多个“程序”处理不同的数据。需要确定是哪一个程序：
    pid = tl.program_id(axis=0)  # 使用 1D 网格，因此轴为 0。

    # 该程序将处理相对初始数据偏移的输入。
    # 例如，如果有一个长度为 256, 块大小为 64 的向量，程序将各自访问 [0:64, 64:128, 128:192, 192:256] 的元素。
    # 注意 offsets 是指针列表：
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 创建 mask 防止内存操作超出边界访问。
    mask = offsets < n_elements

    # 从 DRAM 加载 x 和 y，如果输入不是块大小的整数倍，则屏蔽掉任何多余的元素。
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    # 将 x + y 写回 DRAM。
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    """加法实例函数：用适当的 grid/block sizes 将上述内核加入队列"""
    output = torch.empty_like(x)
    assert x.device == y.device == output.device == DEVICE
    n_elements = output.numel()

    # SPMD 启动网格表示并行运行的内核实例的数量。
    # 它类似于 CUDA 启动网格。它可以是 Tuple[int]，也可以是 Callable(metaparameters) -> Tuple[int]。
    # 在这种情况下，使用 1D 网格，其中大小是块的数量：
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # 注意：
    #  - 每个 torch.tensor 对象都会隐式转换为其第一个元素的指针。
    #  - `triton.jit` 函数可以通过启动网格索引来获得可调用的 GPU 内核。
    #  - 不要忘记以关键字参数传递元参数。
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # 返回 z 的句柄，但由于 `torch.cuda.synchronize()` 尚未被调用，此时内核仍在异步运行。
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 100000
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    time_1 = time.time()
    output_torch = x + y
    time_2 = time.time()
    output_triton = add(x, y)
    time_3 = time.time()

    print(torch.max(torch.abs(output_torch - output_triton)) == 0)  # tensor(True)

    print(time_2 - time_1)  # 0.012052774429321289
    print(time_3 - time_2)  # 1.1916394233703613
