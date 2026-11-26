import torch


def assert_close(actual, expected, data_type, *, source=None):
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
    source_prefix = f"source={source}, " if source is not None else ""
    report = (
        f"{source_prefix}shape={tuple(actual.shape)}, dtype={data_type}, "
        f"max_abs_error={max_abs_error:.6e}, max_rel_error={max_rel_error}"
    )
    print(report)
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol,
        msg=lambda message: f"{report}\n{message}",
    )
