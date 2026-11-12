import pytest
from references.rmsnorm import rmsnorm_reference
import torch

def assert_close(actual, expected, data_type, source):
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
        f"source={source}, shape={tuple(actual.shape)}, dtype={data_type}, "
        f"max_abs_error={max_abs_error:.6e}, max_rel_error={max_rel_error}"
    )
    print(report)
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol,
        msg=lambda message: f"{report}\n{message}",
    )


def rmsnorm_fp64_oracle(x, weight, eps):
    # 从目标 dtype 的实际输入出发，所有中间计算（包括 weight）均用 FP64。
    x64 = x.double()
    mean_square = x64.square().sum(dim=-1, keepdim=True) / x.shape[-1]
    return (x64 / torch.sqrt(mean_square + eps) * weight.double()).to(x.dtype)


@pytest.mark.parametrize("shape, message", [
    ((0, 4), "x's first dimension must be greater than 0"),
    ((4, 0), "The reduction dimension must be greater than 0"),
    ((0, 0), "x's first dimension must be greater than 0"),
])
def test_rmsnorm_reference_empty_input(shape, message):
    x = torch.empty(shape, dtype=torch.float32)
    weight = torch.empty(shape[-1], dtype=torch.float32)
    with pytest.raises(ValueError, match=f"^{message}$"):
        rmsnorm_reference(x, weight)

@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (1, 128), (1, 4096), (16, 768), (128, 4096), (256, 8192)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rmsnorm_reference(shape, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, generator=generator)
    weight = torch.randn(shape[-1], dtype=dtype, generator=generator)
    eps = 1e-6
    actual = rmsnorm_reference(x, weight, eps)
    expected_pytorch = torch.nn.functional.rms_norm(
        x, (x.shape[-1],), weight=weight, eps=eps
    )
    assert_close(actual, expected_pytorch, dtype, source="pytorch")
    # 小 shape 同时验证独立数学公式；大 shape 保留官方算子对照。
    if x.numel() <= 16384:
        expected_oracle = rmsnorm_fp64_oracle(x, weight, eps)
        assert_close(actual, expected_oracle, dtype, source="fp64_oracle")
    

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "case", ["zero", "constant", "tiny", "large", "mixed_sign"]
)
def test_rmsnorm_reference_edge_values(case, dtype):
    eps = 1e-6
    if case == "zero":
        x = torch.zeros((2, 7), dtype=dtype)
    elif case == "constant":
        # 不同长度验证求均值时的 reduction 除数。
        x = torch.full((2, 7), 3.0, dtype=dtype)
    elif case == "tiny":
        # FP16 中直接平方会下溢；同时覆盖 epsilon 主导和平方项不可忽略的情况。
        x = torch.tensor([[1e-4, -1e-4, 0, 1e-4],
                          [1e-3, -1e-3, 0, 1e-3]], dtype=dtype)
    elif case == "large":
        # 输入本身可由 FP16 表示，但平方、求和会超出 FP16 范围。
        x = torch.full((2, 128), 1000.0, dtype=dtype)
    else:
        x = torch.tensor([[-3, -1, 0, 1, 3],
                          [3, 1, 0, -1, -3]], dtype=dtype)

    weight = torch.ones(x.shape[-1], dtype=dtype)
    actual = rmsnorm_reference(x, weight, eps)

    expected_pytorch = torch.nn.functional.rms_norm(
        x, (x.shape[-1],), weight=weight, eps=eps
    )
    expected_oracle = rmsnorm_fp64_oracle(x, weight, eps)
    assert torch.isfinite(actual).all()
    assert_close(actual, expected_pytorch, dtype, source="pytorch")
    assert_close(actual, expected_oracle, dtype, source="fp64_oracle")

    if case == "zero":
        # epsilon 保证分母非零，全零输入应精确输出零。
        torch.testing.assert_close(actual, torch.zeros_like(x), atol=0, rtol=0)

    if case == "mixed_sign":
        # 正权重下，归一化必须保留正负号和零。
        torch.testing.assert_close(actual.sign(), x.sign(), atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(), (3,), (1, 3), (5,)])
def test_rmsnorm_reference_invalid_weight_shape(shape):
    with pytest.raises(ValueError, match="weight should be"):
        rmsnorm_reference(torch.ones(2, 4), torch.ones(shape))


@pytest.mark.parametrize("x_dtype, weight_dtype", [
    (torch.float32, torch.float16),
    (torch.float16, torch.bfloat16),
    (torch.bfloat16, torch.float32),
])
def test_rmsnorm_reference_mismatched_dtype(x_dtype, weight_dtype):
    with pytest.raises(TypeError, match="same dtype"):
        rmsnorm_reference(torch.ones(2, 4, dtype=x_dtype),
                          torch.ones(4, dtype=weight_dtype))


@pytest.mark.parametrize("x_device, weight_device", [("cpu", "meta"), ("meta", "cpu")])
def test_rmsnorm_reference_mismatched_device(x_device, weight_device):
    # meta 张量不分配真实存储，可在没有 GPU 的环境中验证设备校验。
    with pytest.raises(ValueError, match="same device"):
        rmsnorm_reference(torch.ones(2, 4, device=x_device),
                          torch.ones(4, device=weight_device))


@pytest.mark.parametrize("target", ["x", "weight"])
def test_rmsnorm_reference_noncontiguous(target):
    x = torch.ones(2, 4)
    weight = torch.ones(4)
    if target == "x":
        x = torch.ones(4, 2).t()
        assert not x.is_contiguous()
    else:
        weight = torch.ones(8)[::2]
        assert not weight.is_contiguous()
    with pytest.raises(ValueError, match="must be contiguous"):
        rmsnorm_reference(x, weight)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64, torch.bool, torch.complex64])
@pytest.mark.parametrize("target", ["x", "weight"])
def test_rmsnorm_reference_unsupported_dtype(target, dtype):
    x = torch.ones(2, 4, dtype=dtype if target == "x" else torch.float32)
    weight = torch.ones(4, dtype=dtype if target == "weight" else torch.float32)
    with pytest.raises(TypeError, match=rf"{target}\.dtype=.*is not supported"):
        rmsnorm_reference(x, weight)


@pytest.mark.parametrize("shape", [(), (4,), (1, 2, 4)])
def test_rmsnorm_reference_invalid_input_rank(shape):
    with pytest.raises(ValueError, match="must be a 2D tensor"):
        rmsnorm_reference(torch.ones(shape), torch.ones(4))


@pytest.mark.parametrize("target", ["x", "weight"])
def test_rmsnorm_reference_non_tensor(target):
    x = [[1.0] * 4] if target == "x" else torch.ones(2, 4)
    weight = [1.0] * 4 if target == "weight" else torch.ones(4)
    with pytest.raises(TypeError, match=f"{target} must be torch.Tensor"):
        rmsnorm_reference(x, weight)


@pytest.mark.parametrize("eps", [0, 0.0, -1, -1e-6, float("nan"), float("inf"), -float("inf")])
def test_rmsnorm_reference_invalid_eps(eps):
    with pytest.raises(ValueError, match="^eps must be finite and greater than 0$"):
        rmsnorm_reference(torch.ones(2, 4), torch.ones(4), eps)


@pytest.mark.parametrize("eps", [None, "1e-6", True, False, 1 + 0j, [], torch.tensor(1e-6)])
def test_rmsnorm_reference_invalid_eps_type(eps):
    with pytest.raises(TypeError, match="^eps must be a real number$"):
        rmsnorm_reference(torch.ones(2, 4), torch.ones(4), eps)


@pytest.mark.parametrize("eps", [1, 1e-6])
def test_rmsnorm_reference_valid_eps(eps):
    x = torch.ones(2, 4)
    actual = rmsnorm_reference(x, torch.ones(4), eps)
    expected = torch.full_like(x, 1 / (1 + eps) ** 0.5)
    torch.testing.assert_close(actual, expected)

def test():
    x = torch.tensor([1e-4, 1e-4, 1e-4, 1e-4], dtype=torch.float16)
    y = x.square()
    print(y)
    x_fp32 = x.to(torch.float32)
    y_fp32 = x_fp32.square()
    print(y_fp32)

