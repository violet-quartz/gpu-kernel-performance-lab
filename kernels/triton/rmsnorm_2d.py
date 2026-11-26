import torch
from common.validation import validate_rmsnorm_inputs, validate_rmsnorm_output

import triton
import triton.language as tl

@triton.jit
def rmsnorm_2d_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,
    eps,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    rows = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0).to(tl.float32) # [BLOCK_M, BLOCK_N]
    weight = tl.load(weight_ptr + cols, mask=cols < N, other=0).to(tl.float32) # [BLOCK_N]
    mean_square = tl.sum(x * x, axis=1) / N # [BLOCK_M, ]
    inv_rms = tl.rsqrt(mean_square + eps)
    output = x * inv_rms[:, None] * weight[None, :]
    tl.store(output_ptr + rows[:, None] * N + cols[None, :], output, mask=mask)
    

def rmsnorm_triton_2d_into(
    x:torch.Tensor, # [M, N]
    weight: torch.Tensor, # [N]
    output: torch.Tensor, # [M, N]
    eps: float = 1e-6,
    num_warps: int = 4,
    validate: bool = True):
    if validate:
        validate_rmsnorm_inputs(x, weight, eps)
        if not x.is_cuda:
            raise ValueError("x must be on CUDA")
        validate_rmsnorm_output(output, x, weight)
    M, N = x.shape
    BLOCK_M = 2
    BLOCK_N = triton.next_power_of_2(N)
    grid = (triton.cdiv(M, BLOCK_M), )
    rmsnorm_2d_kernel[grid](x, output, weight, eps, M, N, 
                         BLOCK_M=BLOCK_M,
                         BLOCK_N=BLOCK_N,
                         num_warps=num_warps)

def rmsnorm_triton_2d(
    x: torch.Tensor, # [M, N]
    weight: torch.Tensor, # [N]
    eps: float = 1e-6,
    num_warps: int = 4
) -> torch.Tensor:
    output = torch.empty_like(x)
    rmsnorm_triton_2d_into(x, weight, output, eps, num_warps)
    return output
