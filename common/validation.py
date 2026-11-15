import torch
import math

def overlaps_contiguous(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.device != b.device:
        return False

    if a.numel() == 0 or b.numel() == 0:
        return False

    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Only contiguous tensors are supported")

    a_start = a.data_ptr()
    a_end = a_start + a.numel() * a.element_size()

    b_start = b.data_ptr()
    b_end = b_start + b.numel() * b.element_size()

    return a_start < b_end and b_start < a_end
    
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
    if overlaps_contiguous(x, weight):
        raise ValueError("Overlaps are not allowed between x and weight")
    if x.shape[0] == 0:
        raise ValueError("x's first dimension must be greater than 0")
    if x.shape[-1] == 0:
        raise ValueError("The reduction dimension must be greater than 0")
    
    if not isinstance(eps, (int, float)) or isinstance(eps, bool):
        raise TypeError("eps must be a real number")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and greater than 0")
    
def validate_rmsnorm_output(output, x, weight):
    if output.shape != x.shape:
        raise ValueError("output should have the same shape with x")
    if output.dtype != x.dtype:
        raise ValueError("output should have the same dtype with x")
    if output.device != x.device:
        raise ValueError("output and x should be in the same device")
    if not output.is_contiguous():
        raise ValueError("output should be contiguous")
    if overlaps_contiguous(output, x):
        raise ValueError("Overlaps are not allowed between x and output")
    if overlaps_contiguous(output, weight):
        raise ValueError("Overlaps are not allowed between weight and output")
    