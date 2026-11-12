import torch
import math

def validate_rmsnorm_inputs(x, weight, eps):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be torch.Tensor")
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"x must be a 2D tensor. Currently it's {tuple(x.shape)}")
    if weight.shape != (x.shape[1],):
        raise ValueError( f"weight should be ({x.shape[1]},), actually it's {tuple(weight.shape)}")
    
    supported = (torch.float32, torch.float16, torch.bfloat16)
    if x.dtype not in supported:
        raise TypeError(f"x.dtype={x.dtype} is not supported")
    if weight.dtype not in supported:
        raise TypeError(f"weight.dtype={weight.dtype} is not supported")
    if weight.dtype != x.dtype:
        raise TypeError("weight should be the same dtype with x")

    if weight.device != x.device:
        raise ValueError("weight and x should be in the same device")

    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    if x.shape[0] == 0:
        raise ValueError("x's first dimension must be greater than 0")
    if x.shape[-1] == 0:
        raise ValueError("The reduction dimension must be greater than 0")
    
    if not isinstance(eps, (int, float)) or isinstance(eps, bool):
        raise TypeError("eps must be a real number")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and greater than 0")
    