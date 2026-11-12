from common.validation import validate_rmsnorm_inputs
import torch


def rmsnorm_reference(
    x: torch.Tensor, # [M, N]
    weight: torch.Tensor, # [N]
    eps: float = 1e-6,
) -> torch.Tensor:
    validate_rmsnorm_inputs(x, weight, eps)
    dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    rms = torch.sqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps) # [M, 1]
    y = x_fp32 / rms * weight_fp32
    y = y.to(dtype)
    return y