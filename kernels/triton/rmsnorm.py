import torch
from common.validation import validate_rmsnorm_inputs

import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,
    eps,
    cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols
    x = tl.load(x_ptr + pid * cols + offsets, mask=mask, other=0).to(tl.float32) # (BLOCK_SIZE,)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / cols
    inv_rms = tl.rsqrt(mean_square + eps)
    output = x * inv_rms * weight
    tl.store(output_ptr + pid * cols + offsets, output, mask=mask)
    

def rmsnorm_triton(
    x: torch.Tensor, # [M, N]
    weight: torch.Tensor, # [N]
    eps: float = 1e-6
) -> torch.Tensor:
    validate_rmsnorm_inputs(x, weight, eps)
    if not x.is_cuda:
        raise ValueError("x must be on CUDA")
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    rmsnorm_kernel[grid](x, output, weight, eps, N, BLOCK_SIZE)
    return output
