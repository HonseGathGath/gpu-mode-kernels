#!POPCORN leaderboard vectorsum_v2
#!POPCORN gpu A100

import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048 }, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def sum_kernel(
    x_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Parallel reduction kernel that sums elements in chunks.
    Each thread block reduces BLOCK_SIZE elements.
    """
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0).to(tl.int64)
    n_elements = n_elements.to(tl.int64)

    acc0 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    i = pid * BLOCK_SIZE
    stride = num_programs * BLOCK_SIZE
    big_stride = 4 * stride

    while i + 3 * stride + BLOCK_SIZE <= n_elements:
        off0 = i + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        off1 = off0 + stride
        off2 = off0 + 2 * stride
        off3 = off0 + 3 * stride
        acc0 += tl.load(x_ptr + off0)
        acc1 += tl.load(x_ptr + off1)
        acc2 += tl.load(x_ptr + off2)
        acc3 += tl.load(x_ptr + off3)
        i += big_stride

    while i < n_elements:
        offs = i + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc0 += x
        i += stride

    total = acc0 + acc1 + acc2 + acc3
    block_sum = tl.sum(total, axis=0)
    tl.store(partial_ptr + pid, block_sum)

def _custom_kernel(data: input_t) -> output_t:
    """
    Performs parallel reduction to compute sum of all elements.
    Args:
        data: Input tensor to be reduced
    Returns:
        Tensor containing the sum of all elements
    """
    data, output = data
    n_elements = data.numel()

    # Configure kernel
    MAX_BLOCKS = 216
    NUM_BLOCKS = min(MAX_BLOCKS, max(1, triton.cdiv(n_elements, 1024)))
    # grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    partials = torch.empty(NUM_BLOCKS, device=data.device, dtype=torch.float32)

    # Launch kernel
    sum_kernel[(NUM_BLOCKS,)](
        data,
        partials,
        n_elements,
    )

    output[0] = partials.sum()

    return output[0]


# Compile the kernel for better performance
# custom_kernel = torch.compile(_custom_kernel, mode="reduce-overhead")
custom_kernel =_custom_kernel
