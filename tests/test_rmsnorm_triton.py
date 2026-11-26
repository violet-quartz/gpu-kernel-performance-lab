import pytest
import torch
from references.rmsnorm import rmsnorm_reference
from kernels.triton.rmsnorm import rmsnorm_triton, rmsnorm_triton_into
from tests.rmsnorm_assertions import assert_close


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rmsnorm_triton_into(dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    x = torch.randn((16, 768), dtype=dtype, generator=generator).cuda()
    weight = torch.randn(x.shape[-1], dtype=dtype, generator=generator).cuda()
    output = torch.empty_like(x)

    rmsnorm_triton_into(x, weight, output)

    reference = rmsnorm_reference(x, weight)
    assert_close(output, reference, dtype)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device


@pytest.mark.parametrize(
    "output, message",
    [
        (lambda x: torch.empty((2, 5), device="cuda"), "output should have the same shape"),
        (
            lambda x: torch.empty(x.shape, dtype=torch.float64, device="cuda"),
            "output should have the same dtype",
        ),
        (
            lambda x: torch.empty(x.shape, dtype=x.dtype, device="cpu"),
            "output and x should be in the same device",
        ),
    ],
)
def test_rmsnorm_triton_into_invalid_output_contract(output, message):
    x = torch.randn((2, 4), device="cuda")
    weight = torch.ones(4, device="cuda")

    with pytest.raises(ValueError, match=message):
        rmsnorm_triton_into(x, weight, output(x))


@pytest.mark.parametrize("overlap", ["complete", "partial", "view"])
def test_rmsnorm_triton_into_rejects_output_overlapping_x(overlap):
    if overlap == "complete":
        x = torch.randn((2, 4), device="cuda")
        output = x
    else:
        storage = torch.randn(10, device="cuda")
        x = storage[:8].view(2, 4)
        if overlap == "partial":
            output = storage[2:10].view(2, 4)
        else:
            output = storage.as_strided((2, 4), (4, 1), storage_offset=1)
    weight = torch.ones(4, device="cuda")

    with pytest.raises(ValueError, match="Overlaps are not allowed between x and output"):
        rmsnorm_triton_into(x, weight, output)
