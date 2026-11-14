import triton.testing
import torch
from references.rmsnorm import rmsnorm_reference
from kernels.triton.rmsnorm import rmsnorm_triton

shapes = [(1, 4096), (16, 4096), (128, 4096), (1024, 4096), (128, 8192)]
data_types = [torch.float16, torch.bfloat16]
fns = [rmsnorm_reference, rmsnorm_triton]
eps = 1e-6
return_mode = 'median'

generator = torch.Generator(device="cpu")
generator.manual_seed(42)
for shape in shapes:
    for dtype in data_types:
        x = torch.randn(shape, generator=generator, dtype=dtype).to('cuda')
        weight = torch.randn(shape[-1], generator=generator, dtype=dtype).to('cuda')
        print(f'Median latency of Shape: {shape}, dtype: {dtype}')
        for fn in fns:
            latency_ms = triton.testing.do_bench(
                lambda: fns(x, weight, eps),
                warmup=100,
                rep=500,
                return_mode=return_mode
            )
            latency_us = latency_ms * 1000
            print(f'{fn.__name__}: {latency_us:.2f} us')
        # test torch.nn.functional.rms_norm
        latency_ms = triton.tesing.do_bench(
            lambda: torch.nn.functional.rms_norm(x, shape[-1], weight=weight, eps=eps),
            warmup=100,
            rep=50,
            return_mode=return_mode
        )
        latency_us = latency_ms * 1000
        print(f'torch.nn.functional.rms_norm: {latency_us:.2f} us')

