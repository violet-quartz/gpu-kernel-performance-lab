import pytest
import torch
from references.rmsnorm import rmsnorm_reference
from kernels.triton.rmsnorm_2d import rmsnorm_triton_2d, rmsnorm_triton_2d_into
from tests.rmsnorm_assertions import assert_close


@pytest.mark.parametrize("shape", [(1, 4096), (3, 4096), (128, 4096), (4096, 4096), (16, 768)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rmsnorm_triton(shape, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, generator=generator).cuda()
    weight = torch.randn(shape[-1], dtype=dtype, generator=generator).cuda()
    eps=1e-6
    triton_res = rmsnorm_triton_2d(x, weight, eps)
    reference_res = rmsnorm_reference(x, weight, eps)
    assert_close(triton_res, reference_res, dtype)
    assert triton_res.shape == x.shape
    assert triton_res.dtype == x.dtype
    assert triton_res.device == x.device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rmsnorm_triton_into(dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    x = torch.randn((16, 768), dtype=dtype, generator=generator).cuda()
    weight = torch.randn(x.shape[-1], dtype=dtype, generator=generator).cuda()
    output = torch.empty_like(x)

    rmsnorm_triton_2d_into(x, weight, output)

    reference = rmsnorm_reference(x, weight)
    assert_close(output, reference, dtype)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device
