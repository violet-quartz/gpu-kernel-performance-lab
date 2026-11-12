import pytest
import torch
from references.rmsnorm import rmsnorm_reference
from kernels.triton.rmsnorm import rmsnorm_triton

def assert_close(actual, expected, data_type):
    tolerances = {
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (2e-2, 2e-2),
        torch.float32: (1e-5, 1e-5),
    }
    atol, rtol = tolerances[data_type]
    # 用 FP64 统计，避免低精度减法掩盖实际误差。
    abs_error = (actual.double() - expected.double()).abs()
    expected_abs = expected.double().abs()
    # 相对误差为 |actual - expected| / |expected|，排除分母为零的元素。
    nonzero = expected_abs != 0
    max_abs_error = abs_error.max().item()
    max_rel_error = (
        f"{(abs_error[nonzero] / expected_abs[nonzero]).max().item():.6e}"
        if nonzero.any() else "N/A"
    )
    report = (
        f"shape={tuple(actual.shape)}, dtype={data_type}, "
        f"max_abs_error={max_abs_error:.6e}, max_rel_error={max_rel_error}"
    )
    print(report)
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol,
        msg=lambda message: f"{report}\n{message}",
    )

@pytest.mark.parametrize("shape", [(1, 7), (16, 768), (128, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rmsnorm_triton(shape, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, generator=generator).cuda()
    weight = torch.randn(shape[-1], dtype=dtype, generator=generator).cuda()
    eps=1e-6
    triton_res = rmsnorm_triton(x, weight, eps)
    reference_res = rmsnorm_reference(x, weight, eps)
    assert_close(triton_res, reference_res, dtype)
    assert triton_res.shape == x.shape
    assert triton_res.dtype == x.dtype
    assert triton_res.device == x.device
