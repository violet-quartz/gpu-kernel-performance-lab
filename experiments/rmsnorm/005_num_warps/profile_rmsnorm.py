import argparse

import torch
from kernels.triton.rmsnorm import rmsnorm_triton_into
import triton.testing

N = 4096
data_type = torch.float16
eps = 1e-6
return_mode = "median"

def main():
    parser = argparse.ArgumentParser(description="Profile the Triton RMSNorm kernel.")
    parser.add_argument(
        "--M",
        type=int,
        default=128
    )
    parser.add_argument(
        "--num_warps",
        type=int,
        default=4
    )
    args = parser.parse_args()
    M = args.M
    num_warps = args.num_warps
    if M <= 0:
        parser.error("--M must be positive")
    if num_warps <= 0:
        parser.error("--num_warps must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    x = torch.randn((M, N), generator=generator, dtype=data_type).to("cuda")
    weight = torch.randn((N,), generator=generator, dtype=data_type).to("cuda")
    output = torch.empty_like(x)

    # JIT warmup
    rmsnorm_triton_into(
        x, weight, output,
        eps=eps,
        num_warps=num_warps,
        validate=False,
    )
    torch.cuda.synchronize()

    # CUDA Graph bench
    latency_us = triton.testing.do_bench_cudagraph(
                                lambda: rmsnorm_triton_into(x, weight, output, eps=eps, num_warps=num_warps, validate=False),
                                rep=500,
                                return_mode=return_mode
                            ) * 1000
    print(f'shape=({M}, 4096), num_warps={num_warps}, latency median={latency_us} us.')

if __name__ == "__main__":
    main()
    